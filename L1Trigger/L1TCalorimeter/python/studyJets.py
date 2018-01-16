#!/usr/bin/env python

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



def getJetProperties(jetSeed,etaIn,etaOut):

    jetSize = 0
    etaInSize = 0
    etaOutSize = 0

    for ring in xrange(jetSeed-etaOut,jetSeed+etaIn+1):

        if ring < 1:
            ring = abs(ring-1)

        if ring >= len(towerEtaWidths):
            break

        if ring < jetSeed:
            etaInSize += towerEtaWidths[ring]
        if ring > jetSeed:
            etaOutSize += towerEtaWidths[ring]

    jetSize = (etaInSize + etaOutSize + towerEtaWidths[jetSeed])/0.8
    seedCent = etaOutSize/etaInSize

    jetProps = [jetSize, seedCent]

    return jetProps


def printJetProperties(etaRange):

    print "Size  \  eta\t",
    for seedEta in etaRange:
        if(seedEta<29):
            print str(seedEta)+"\t\t",
        else:
            print str(seedEta+1)+"\t\t",
    print

    for size in jetEtaSizes:

        if size%2:
            print "   9x"+str(size)+("\t"),
            for seedEta in etaRange:
                print("\t"),
                etaInOut = (size-1)/2
                jetProps = getJetProperties(seedEta,etaInOut,etaInOut)
                print("%.2f / %.2f" %(jetProps[0],jetProps[1])),
            print

        else:
            print "   9x"+str(size)+" out",
            for seedEta in etaRange:
                print("\t"),
                etaIn = size/2-1
                etaOut = size/2
                jetProps = getJetProperties(seedEta, etaIn, etaOut)
                print("%.2f / %.2f" %(jetProps[0],jetProps[1])),
            print
            print "   9x"+str(size)+" in",
            for seedEta in etaRange:
                print("\t"),
                etaIn = size/2
                etaOut = size/2-1
                jetProps = getJetProperties(seedEta, etaIn, etaOut)
                print("%.2f / %.2f" %(jetProps[0],jetProps[1])),
            print



print "\n"
print "=============================================================================================================================================================================================================================="
print "Eta-dependence of jet sizes & seed centrality"
print "First number  = jet diameter in eta, normalised to 0.8 (offline)"
print "Second number = jet size on inside of jet seed / jet size outside of jet seed"
print "Ideally, best jet choice is where both numbers are closest to unity!"
print "9xN out = larger area on side of jet further from beam pipe"
print "9xN in  = larger area on side of jet closer to beam pipe"
print "=============================================================================================================================================================================================================================="


beEtaRange = xrange(16,29)
hfEtaRange = xrange(29,41)


jetEtaSizes = [9,8,7,6,5,4,3]


print("\n BE \n")
printJetProperties(beEtaRange)

print("\n\n HF \n")
printJetProperties(hfEtaRange)

print("\n")
print "=============================================================================================================================================================================================================================="
print("\n\n")
