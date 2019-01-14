#! /usr/bin/python -O
# -*- coding: utf-8 -*-
# ----------------------------------------------------------
# Entraînement d'un IBM model 1, à partir d'un corpus aligné
# = estimations successives des paramètres t(wf | wf)
# ----------------------------------------------------------
import datetime
import re
import os
import sys
import argparse
from collections import defaultdict
from functools import reduce
import operator
import random
import pickle
import time
import pandas as pd


class IBM1:

    def __init__(self, src_lang_file, target_lang_file, maxlength, maxnbsent):

        # le corpus aligné en phrases
        # = une liste de paires de phrases tokenisées (langue cible (E) / langue source (F))
        #   [ [ ['nullword', 'a', 'sentence'],       ['une', 'phrase']         ],
        #     [ ['nullword', 'another', 'sentence'], ['une', 'autre', 'phrase']  ],
        #   ...
        #   ]
        self.sentences = []

        # --------------------------------------
        # Gestion correspondance mot<->id
        # --------------------------------------
        # Pour chaque mot (de langue E ou F)
        # on utilise un id entier au lieu de la chaîne de caractères
        # Ces id sont incrémentés à chaque fois qu'un nouveau mot est lu
        # Pour la langue E, l'id 0 est réservé au *mot null*
        # et pour chaque phrase e, on stocke le mot null au début
        # Donc, dans sentences, on stocke des id
        #   [ [ [0, 1, 2],       [0, 1]         ],
        #     [ [0, 3, 2],       [0, 2, 1]  ],
        #   ...
        #   ]
        # et word2id = {E:{'nullword':0, 'a':1, 'sentence':2, 'another':3},
        #               F:{'une':0, 'phrase':1, 'autre':2}}
        # et id2word = {E:['nullword','a','sentence','another'],
        #               F:['une','phrase','autre']}

        self.word2id = {'E': {'nullword': 0},
                        'F': {}}
        self.id2word = {'E': ['nullword'],
                        'F': []}
        self.nbword = {'E': 1,
                       'F': 0}
        self.train_time = time.time()

        # chargement des fichiers alignés dans la structure self.sentences
        self.load_sentence_aligned_files(src_lang_file, target_lang_file, maxlength, maxnbsent)

    def get_id(self, word, lg):
        """ pour un couple mot , langue (E ou F)
        rend l'identifiant correspondant
        après l'avoir créé si nécessaire """

        if word not in self.word2id[lg]:
            self.word2id[lg][word] = self.nbword[lg]
            self.id2word[lg].append(word)
            self.nbword[lg] = self.nbword[lg] + 1

        return self.word2id[lg][word]

    def get_word(self, id, lg):
        """ pour un couple id, langue (E ou F)
        rend le mot(=token) correspondant """
        return self.id2word[lg][id]

    # ----------------------------------------
    # Normalisation
    # ----------------------------------------
    def normalize_and_split(self, str):
        # on passe toutes les chaines en minuscules (trop brutal!!)
        str = re.sub(r'[0-9]+', 'NUM', str.lower())
        str.strip()
        return str.split(' ')

    # Chargement des phrases alignées dans la structure sentences
    # attention on ne stocke pas systématiquement le mot null
    def load_sentence_aligned_files(self, src_file, target_file, maxlength, maxnbsent):
        src_stream = open(src_file, 'rU', encoding='utf8')
        target_stream = open(target_file, 'rU', encoding='utf8')
        totalnbsent = 0
        for src_line in src_stream:
            target_line = target_stream.readline()
            totalnbsent += 1
            if not target_line:
                exit('Les fichiers source et cible n\'ont pas le même nombre de lignes!')
            # e et f normalisées (minusculisées, nombres...) et stockées chacune comme liste de tokens
            f = self.normalize_and_split(src_line[:-1])
            e = self.normalize_and_split(target_line[:-1])
            if len(f) > maxlength or len(e) > maxlength:
                continue

            # e_id et f_id = listes des ids de tokens
            e_id = [self.get_id(x, 'E') for x in e]
            f_id = [self.get_id(x, 'F') for x in f]
            # on rajoute le mot null (son id est 0) à chaque phrase e,
            # pour ne plus être embêté
            e_id = [0] + e_id
            self.sentences.append([e_id, f_id])
            if maxnbsent > 0 and len(self.sentences) > maxnbsent:
                return
        sys.stderr.write(
            str(len(self.sentences)) + ' phrases chargees parmi les ' + str(totalnbsent) + ' phrases disponibles\n')

    # ----------------------------------------
    # INITIALISATION des T et C
    # ----------------------------------------
    # Init ne fait que construires les structures vides
    #
    # Pour T (et C), on ne stocke pas toutes les paires we/wf
    # On stocke pour chque we, uniquement les wf pertinents
    # (= ceux qui apparaissent au moins une fois dans la même phrase que we)
    # Cela est obtenu par construction, en parcourant les paires alignees
    # Structure :
    # T est un tableau de dict
    # -- l'indice représente l'id de mot E
    # -- chaque valeur dans le tableau =
    #    = 1 dictionnaire
    #        clé = id de mot F
    #        valeur T(wf | we)
    # C est un tableau de dict, avec le même principe:
    # -- l'indice représente l'id de mot E
    # -- chaque valeur dans le tableau =
    #    = 1 dictionnaire
    #        clé = id de mot F
    #        valeur compte probabilisé C(wf | we)

    def init_T_C(self):
        # pour chaque indice de mot de langue E,
        # on stocke un dictionnaire vide pour l'instant  {}
        # rem : pour T, a l'initialisation, on devrait mettre partout
        # la valeur uniform = 1/nb de mots F = 1/nbword['F']
        # MAIS c'est inutile de le stocker, voir get_T
        self.T = [defaultdict(int) for x in range(self.nbword['E'])]
        self.C = [defaultdict(int) for x in range(self.nbword['E'])]

    # get_T a 2 fonctionnements
    # si is_init est vrai (= pdt la premiere iteration)
    #    -> rend la valeur uniform
    # sinon,
    # si T[we][wf] est connu -> on rend sa valeur
    # sinon, cela signifie par construction que t(wf | we) vaut 0
    # (= we et wf ne sont jamais dans la meme paire de phrases)
    def get_T(self, we, wf, is_init):
        if is_init:
            return 1 / self.nbword['E']
        if wf in self.T[we]:
            return self.T[we][wf]
        return 0

    def get_C(self, we, wf):
        return self.C[we][wf]


# ----------------------------------------
# MAIN
# ----------------------------------------
usage = sys.argv[0] + """ [OPTIONS] FICHIER_LANG_SOURCE FICHIER_LANG_CIBLE

        Entraînement d'un modèle de traduction IBM 1
        Les deux fichiers sont supposés être alignés en phrases : une phrase par ligne pour chacun
        """

# ----------------------------------------
# Récupération options
# ----------------------------------------
# parser = argparse.ArgumentParser(usage=usage)
# parser.add_argument('src_corpus_file', help='Phrases en langue source, tokenisées')
# parser.add_argument('target_corpus_file', help='Phrases alignées avec src_corpus_file, en langue cible, tokenisées')
# parser.add_argument('-i', '--iternb', type=int, default=3,
#                     help='Nombre d\'itérations de l\'algorithme d\'estimation (defaut = 3)')
# parser.add_argument('-l', '--maxlength', type=int, default=30,
#                     help='Longueur maximale des phrases dans les deux langues (Default=30)')
# parser.add_argument('-n', '--maxnbsent', type=int, default=0,
#                     help='Nb max de couples de phrases à utiliser (Defaut=0 : pas de limite). A utiliser en phase de test pour accélérer le debug')
# args = parser.parse_args()

# ----------------------------------------
# Algo
# ----------------------------------------
# une instance de IBM1 => chargement des corpus alignés
ibm1 = IBM1('ep-08-04-fr.filt.tok', 'ep-08-04-en.filt.tok', 30, 0)

# initialisation des structures de données pour membres T et C
ibm1.init_T_C()


# TODO :
# - algo itératif
# - affichage pour chaque we, des wf les plus probables

def product(iterable):
    return reduce(operator.mul, iterable, 1)


def calcul_C(ibm1, is_init):
    for (e, f) in ibm1.sentences:
        for wf in f:
            denom = sum([ibm1.get_T(we, wf, is_init) for we in e])
            for we in e:
                ibm1.C[we][wf] += ibm1.get_T(we, wf, is_init) / denom


def calcul_T(ibm1):
    for (e, f) in ibm1.sentences:
        for we in e:
            denom = sum(ibm1.get_C(we, wf) for wf in range(ibm1.nbword['F']))
            for wf in f:
                ibm1.T[we][wf] = ibm1.get_C(we, wf) / denom


def calcul_T2(ibm1):
    for (e, f) in ibm1.sentences:
        for wf in f:
            denom = sum(ibm1.get_C(we, wf) for we in e)
            for we in e:
                ibm1.T[we][wf] = ibm1.get_C(we, wf) / denom

# afficher mes résultats sous forme de matrice
# organiser ma matrice (lignes, colonnes)
#       on passe à la ligne quand a augmente
#       a augmente quand b augmente 5 fois
#       b augmente quand le mot fr fait plus de 3 caractères
# afficher mots pertinents
def affiche(ibm1):
    the_list = list()
    a=-1
    b=0
    for we in range(100):
        if b%5==0:
            a+=1
            the_list.append(list())
        stats = ibm1.T[we]
        wf = max(stats.items(), key=operator.itemgetter(1))[0]
        if len(ibm1.get_word(wf,'F'))>3:
            a_list = list()
            a_list.extend([ibm1.get_word(we, 'E'),
                           ibm1.get_word(wf, 'F'),
                           ibm1.T[we][wf]])  # print la clé à la valeur la plus grande
            the_list[a].extend(a_list)
            b+=1
    df=pd.DataFrame(the_list)
    df.style.set_properties(**{'text-align': 'left'})
    print(df)


with open('ibm1', 'rb') as pickle_file:
    ibm1 = pickle.load(pickle_file)
#
# is_init = True
# for i in range(args.iternb):
#     ibm1.C = [defaultdict(int) for x in range(ibm1.nbword['E'])]
#     calcul_C(ibm1, is_init)
#     calcul_T(ibm1)
#     is_init = False
#
# ibm1.train_time = datetime.timedelta(seconds=(ibm1.train_time - time.time()) * -1)
# pickle.dump(ibm1, open('ibm1(1)', 'wb'))

print(ibm1.train_time)

# e = ''
# f = ''
# i = 120
# for we in ibm1.sentences[i][0]:
#     e += ibm1.get_word(we, 'E') + ' '
# for wf in ibm1.sentences[i][1]:
#     f += ibm1.get_word(wf, 'F') + ' '
# print(e)
# print(f)
# print(f.split()[-3], ibm1.get_id(f.split()[-3],'F'))
# print(ibm1.T[ibm1.get_id(e.split()[-4], 'E')][ibm1.get_id(f.split()[-4], 'F')])
affiche(ibm1)

# les_t=[ibm1.T[we][wf] for we in range(len(ibm1.T)) for wf in ibm1.T[we]]
# print(random.sample(les_t,20))
