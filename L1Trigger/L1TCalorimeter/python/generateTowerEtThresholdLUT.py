#!/usr/bin/env python

import os
import sys
import math


###############################################################################################
#  Python script for generating LUT to return tower Et threshold for energy sums              #
#  Input 1: 5 bits - compressed pileup estimate, as used for EG                               #
#  Input 2: 6 bits - abs(ieta) = absolute value of ieta of the trigger tower                  #
#  Tower Et threshold not applied for ieta <= 15                                              # 
#  LUT address input = compressedPileupEstimate << 6 | abs(ieta)                              #
#  Returns 9 bits for tower et threshold                                                      #
#  Author: Aaron Bundock (aaron.*nospamthankyamaam*bundock@cern.ch)                           #
#  Date: 26/04/17                                                                             #  
#                                                                                             #
###############################################################################################

# Run from src/ directory in your checked out CMSSW code!

# open LUT file for writing
if not os.path.isdir(os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data"):
    print(os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data/ directory does not exist.\n"
          "Creating directory now.\n"
          "Remember to do 'git add " + os.environ['LOCALRT'] + "L1Trigger/L1TCalorimeter/data' when committing the new LUT!")
    os.makedirs(os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data")

print "Creating tower Et threshold LUT with filename " + os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data/lut_towEtThresh_2017v6.txt'"
towEtThreshLUTFile = open(os.environ['LOCALRT']+"/src/L1Trigger/L1TCalorimeter/data/lut_towEtThresh_2017v6.txt", "w")


# write header info
towEtThreshLUTFile.write(
    "# address to et sum tower Et threshold LUT\n"
    "# maps 11 bits to 9 bits\n"
    "# 11 bits = (compressedPileupEstimate << 6) | abs(ieta)\n"
    "# compressedPileupEstimate is unsigned 5 bits, abs(ieta) is unsigned 6 bits\n"
    "# data: tower energy threshold returned has 9 bits \n"
    "# anything after # is ignored with the exception of the header\n"
    "# the header is first valid line starting with #<header> versionStr nrBitsAddress nrBitsData </header>\n"
    "#<header> v1 11 9 </header>\n" 

)

# vector of calo tower areas, relative to central barrel areas (0.087 in eta)
# dummy for ieta=0 and excludes ieta=29, since they don't physically exist!

towerAreas = [0.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.03,1.15,1.3,1.48,1.72,2.05,1.72,4.02,
                  3.29,2.01,2.02,2.01,2.02,2.0,2.03,1.99,2.02,2.04,2.00,3.47];
  
etaRange = xrange(0,41) # dummy entry for ieta=0, do not count ieta=29, so count 40 towers
compNTT4Range = xrange(0,32) #  use compressed pileup estimate from EG LUT
addr = 0
printBins = ""

for compNTT4 in compNTT4Range:
    for ieta in etaRange:
        if compNTT4 < 16:
            towEtThresh = int(round(pow(float(towerAreas[ieta]),1.4)*(1/(1+math.exp(-0.07*(ieta))))*(pow(float(compNTT4),2)/100)))
        else:
            towEtThresh = int(round(pow(float(towerAreas[ieta]),1.4)*(1/(1+math.exp(-0.07*(ieta))))*(pow(float(16),2)/100)))
        if ieta > 28:
            towEtThresh -= 2
        if towEtThresh > 12:
            towEtThresh = int(12)
        if ieta < 13 or towEtThresh < 0:
            towEtThresh = 0
        if (addr % 64) == 0:
            printBins = "             # nTT4 = " + str(5*compNTT4) + "-" + str((5*compNTT4)+5) + " ieta = " + str(ieta)  
        elif ieta>28:
            printBins = "             # ieta = " + str(ieta+1)
        else:
            printBins = "             # ieta = " + str(ieta)
        towEtThreshLUTFile.write(
            str(addr) + " " + 
            str(towEtThresh) +
            printBins +
            "\n"
            )
        addr+=1
    if ieta == 40: # dummy to fill 6 bits for eta
        extraCount = 0
        while extraCount < 23:
            towEtThreshLUTFile.write(
                str(addr) + " " + 
                str(0) + 
                " #dummy\n"
                )
            addr+=1
            extraCount +=1

if addr < 2047:
    for addr in xrange(addr,2047):
        towEtThreshLUTFile.write(str(addr) + " " + str(0) + " # dummy\n")
        addr+=1

print "Done. Closing file..."

towEtThreshLUTFile.close()
