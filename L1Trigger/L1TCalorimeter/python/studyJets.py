#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import math


####################################################################################
#  Python script for studying eta dependence of jet sizes and seed centrality      #
#  Author: Aaron Bundock (aaron.*nospamthankyamaam*bundock@cern.ch)                #
#  Date: 15/01/18                                                                  #  
#                                                                                  #
####################################################################################



towerEtaWidths = [0, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087,                 # 0-10
                     0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087, 0.087,                 # 11-20
                     0.090, 0.100, 0.113, 0.129, 0.150, 0.178, 0.150, 0.350,                               # 21-28
                     0.132, 0.175, 0.176, 0.174, 0.176, 0.174, 0.177, 0.173, 0.175, 0.177, 0.173, 0.302]   # 30-41
#0.150, 0.180, 0.170, 0.180, 0.170, 0.180, 0.170, 0.180, 0.180, 0.180, 0.180, 0.290]



def getJetProperties(jetSeed,etaFwd,etaCen):

    jetSize = 0
    etaFwdSize = 0
    etaCenSize = 0

    for ring in xrange(jetSeed-etaCen,jetSeed+etaFwd+1):

        if ring < 1:
            ring = abs(ring-1)

        if ring >= len(towerEtaWidths):
            break

        if ring < jetSeed:
            etaFwdSize += towerEtaWidths[ring]
        if ring > jetSeed:
            etaCenSize += towerEtaWidths[ring]

    jetSize = (etaFwdSize + etaCenSize + towerEtaWidths[jetSeed])/0.8
    seedCent = etaCenSize/etaFwdSize

    jetProps = [jetSize, seedCent]

    return jetProps


def printJetProperties(etaRange):

    print("Size  \  eta\t", end=' ')
    for seedEta in etaRange:
        if(seedEta<29):
            print(str(seedEta)+"\t\t", end=' ')
        else:
            print(str(seedEta+1)+"\t\t", end=' ')
    print()

    for size in jetEtaSizes:

        if size%2:
            print("   9x"+str(size)+("\t"), end=' ')
            for seedEta in etaRange:
                print(("\t"), end=' ')
                etaFwdCen = (size-1)/2
                jetProps = getJetProperties(seedEta,etaFwdCen,etaFwdCen)
                print(("%.2f / %.2f" %(jetProps[0],jetProps[1])), end=' ')
            print()

        else:
            print("   9x"+str(size)+" cen", end=' ')
            for seedEta in etaRange:
                print(("\t"), end=' ')
                etaFwd = size/2-1
                etaCen = size/2
                jetProps = getJetProperties(seedEta, etaFwd, etaCen)
                print(("%.2f / %.2f" %(jetProps[0],jetProps[1])), end=' ')
            print()
            print("   9x"+str(size)+" fwd", end=' ')
            for seedEta in etaRange:
                print(("\t"), end=' ')
                etaFwd = size/2
                etaCen = size/2-1
                jetProps = getJetProperties(seedEta, etaFwd, etaCen)
                print(("%.2f / %.2f" %(jetProps[0],jetProps[1])), end=' ')
            print()



print("\n")
print("==============================================================================================================================================================================================================================")
print("Eta-dependence of jet sizes & seed centrality")
print("First number  = jet diameter in eta, normalised to 0.8 (offline)")
print("Second number = jet size on forward side of jet seed / jet size on central side of jet seed")
print("Ideally, best jet choice is where both numbers are closest to unity!")
print("9xN cen = larger area on side of jet further from beam pipe")
print("9xN fwd  = larger area on side of jet closer to beam pipe")
print("==============================================================================================================================================================================================================================")


beEtaRange = xrange(16,29)
hfEtaRange = xrange(29,41)


jetEtaSizes = [9,8,7,6,5,4,3]


print("\n BE \n")
printJetProperties(beEtaRange)

print("\n\n HF \n")
printJetProperties(hfEtaRange)

print("\n")
print("==============================================================================================================================================================================================================================")
print("\n\n")
