#!/usr/bin/env python

import os
import sys


##############################################################################################
#  Python script for generating LUT to return tower Et threshold for energy sums             #
#  Input 1: 7 bits - nTT4/4 = number of trigger towers within abs(ieta)<=4 divided by 4      #
#  Input 2: 6 bits - abs(ieta) = absolute value of ieta of the trigger tower (41 ieta bins)  # 
#  LUT address input = (nTT4/4) << 6 | abs(ieta)                                             #
#  Author: Aaron Bundock (aaron.*nospamthankyamaam*bundock@cern.ch)                          #
#  Date: 07/04/17                                                                            #  
#                                                                                            #
##############################################################################################

# Run from src/ directory in your checked out CMSSW code!

# open LUT file for writing
if not os.path.isdir(os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data"):
    print(os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data/ directory does not exist.\n"
          "Creating directory now.\n"
          "Remember to do 'git add " + os.environ['LOCALRT'] + "L1Trigger/L1TCalorimeter/data' when committing the new LUT!")
    os.makedirs(os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data")

print "Creating tower Et threshold LUT with filename " + os.environ['LOCALRT'] + "/src/L1Trigger/L1TCalorimeter/data/lut_towEtThresh_2017v1.txt'"
towEtThreshLUTFile = open(os.environ['LOCALRT']+"/src/L1Trigger/L1TCalorimeter/data/lut_towEtThresh_2017v1.txt", "w")


# write header info
towEtThreshLUTFile.write(
    "# address to et sum tower Et threshold LUT\n"
    "# maps 13 bits to 8 bits\n"
    "# 13 bits = (compNTT4<<6) | abs(ieta)\n"
    "# compNTT4 is unsigned 7 bits, abs(ieta) is 6 bits\n"
    "# data: tower energy threshold returned has 9 bits \n"
    "# anything after # is ignored with the exception of the header\n"
    "# the header is first valid line starting with #<header> versionStr nrBitsAddress nrBitsData </header>\n"
    "#<header> v1 13 9 </header>\n" 

)

# vector of calo tower widths in eta, relative to central barrel widths (0.087 in eta)
# excludes ieta=0 and dummy ieta=29

towerEtaWidths = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
                  1.03,1.15,1.3,1.48,1.72,2.05,1.72,4.02,
                  3.29,2.01,2.02,2.01,2.02,2.0,2.03,1.99,2.01,2.03,1.99,3.47];
  
etaRange = xrange(1,41) # 40 towers excluding ieta=29
compNTT4Range = xrange(0,128) #  one bin in nTT4 for every 4 towers. if nTT4 > 512, towEtThresh = 20 (10 GeV) for all eta. 
addr = 0
printBins = ""

for compNTT4 in compNTT4Range:
    for ieta in etaRange:
        towEtThresh = int(round((towerEtaWidths[ieta-1]/4.02)*ieta/5*(compNTT4/10)))
        if towEtThresh > 20:
            towEtThresh = int(20)
        if (addr % 64) == 0:
            printBins = "             # nTT4 = " + str(4*compNTT4) + "-" + str((4*compNTT4)+4) + " ieta = " + str(ieta)  
        else:
            printBins = ""
        towEtThreshLUTFile.write(
            str(addr) + " " + 
            str(towEtThresh) +
            printBins +
            "\n"
            )
        addr+=1
    if ieta == 40: # dummy to fill 6 bits for eta
        extraCount = 0
        while extraCount < 24:
            towEtThreshLUTFile.write(
                str(addr) + " " + 
                str(0) + 
                " #dummy\n"
                )
            addr+=1
            extraCount +=1

if addr < 8192:
    for addr in xrange(addr,8192):
        towEtThreshLUTFile.write(str(addr) + " " + str(0) + " # dummy\n")
        addr+=1

print "Done. Closing file..."

towEtThreshLUTFile.close()
