#!/usr/bin/env python
from __future__ import print_function
import os, sys
import array
import argparse
import RecoLuminosity.LumiDB.LumiConstants as LumiConstants
import re
from math import sqrt

from pprint import pprint
import six

def CalcPileup (deadTable, parameters, luminometer, selBX):
    '''Given a deadtable, will calculate parameters of pileup distribution. Return formatted
    string with LumiSection, LS integrated lumi, RMS of bunch to bunch lumi and pileup.'''

    LumiString = ""
    LumiArray = []

    for lumiSection, deadArray in sorted(six.iteritems(deadTable)):
        numerator = 0
        if luminometer == "HFOC":
            threshold = 8.
        else:
            threshold = 1.2

        numerator     = float (deadArray[1])
        denominator   = float (deadArray[0])
        instLumiArray =        deadArray[2]
        livetime = 1
        if numerator < 0:
            numerator = 0
        if denominator:
            livetime = numerator / denominator

        if lumiSection > 0:
            TotalLumi = 0 
            TotalInt = 0
            TotalInt2 = 0
            TotalWeight = 0
            TotalWeight2 = 0
            FilledXings = 0
            for xing, xingInstLumi, xingDelvLumi in instLumiArray:
                if selBX and xing not in selBX:
                    continue
                xingIntLumi = xingInstLumi * livetime
                mean = xingInstLumi * parameters.orbitLength / parameters.lumiSectionLength
                if mean > 100:
                    if runNumber:
                        print("mean number of pileup events > 100 for run %d, lum %d : m %f l %f" % \
                          (runNumber, lumiSection, mean, xingInstLumi))
                    else:
                        print("mean number of pileup events > 100 for lum %d: m %f l %f" % \
                          (lumiSection, mean, xingInstLumi))
                #print "mean number of pileup events for lum %d: m %f idx %d l %f" % (lumiSection, mean, xing, xingIntLumi)

                if xingInstLumi > threshold:
                    TotalLumi = TotalLumi+xingIntLumi
                    TotalInt+= mean*xingIntLumi
                    FilledXings = FilledXings+1
                    #print "xing inst lumi %f %f %d" % (xingIntLumi,TotalLumi,FilledXings)

            #compute weighted mean, then loop again to get weighted RMS       
            MeanInt = 0
            if TotalLumi >0:
                MeanInt = TotalInt/TotalLumi
            for xing, xingInstLumi, xingDelvlumi in instLumiArray:
                if selBX and xing not in selBX:
                    continue
                if xingInstLumi > threshold:
                    xingIntLumi = xingInstLumi * livetime
                    mean = xingInstLumi * parameters.orbitLength / parameters.lumiSectionLength
                    TotalInt2+= xingIntLumi*(mean-MeanInt)*(mean-MeanInt)
                    TotalWeight+= xingIntLumi
                    TotalWeight2+= xingIntLumi*xingIntLumi

        if ((lumiSection > 0)):
            #print " LS, Total lumi, filled xings %d, %f, %d" %(lumiSection,TotalLumi,FilledXings)
            if FilledXings > 0:
                AveLumi = TotalLumi/FilledXings
            else:
                AveLumi = 0
            RMSLumi = 0
            Denom = TotalWeight*TotalWeight-TotalWeight2
            if TotalLumi > 0 and Denom > 0:
                RMSLumi = sqrt(TotalWeight/(TotalWeight*TotalWeight-TotalWeight2)*TotalInt2)
            LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (lumiSection, TotalLumi, RMSLumi, MeanInt)
            LumiArray.append(lumiSection)
            LumiArray.append(TotalLumi)  # should really weight by total lumi in LS
            LumiArray.append(RMSLumi)
            LumiArray.append(MeanInt)
            lumiX=MeanInt*parameters.lumiSectionLength*FilledXings*(1./parameters.orbitLength)

    #print lumiX
   # if TotalLumi<(lumiX*0.8):
#	print lumiSection
#        print FilledXings
#        print TotalLumi  
#        print lumiX  
#        print numerator
#	print denominator 

    #print FilledXings
    return LumiArray



##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

# modified from the estimatePileup.py script in RecoLuminosity/LumiDB
# 5 Jan, 2012  Mike Hildreth
# The Run 2 version only accepts a csv file from brilcalc as input.

if __name__ == '__main__':
    parameters = LumiConstants.ParametersObject()

    parser = argparse.ArgumentParser(description="Script to estimate average and RMS pileup using the per-bunch luminosity information provided by brilcalc. The output is a JSON file containing a dictionary by runs with one entry per lumi section.")
    parser.add_argument('inputFile', help='CSV input file as produced from brilcalc')
    parser.add_argument('outputFile', help='Name of JSON output file')
    parser.add_argument('-b', '--selBX', help='Comma-separated list of BXs to use (will use all by default)')
    args = parser.parse_args()

    output = args.outputFile

    selBX = set()
    if args.selBX:
        for iBX in args.selBX.split(","):
            try:
                BX=int(iBX)
                selBX.add(BX)
            except:
                print(iBX,"is not an int")
        print("Processing",args.inputFile,"with selected BXs:", sorted(selBX))
    else:
        print("Processing",args.inputFile,"with all BX")

    # The "CSV" file actually is a little complicated, since we also want to split on the colons separating
    # run/fill as well as the spaces separating the per-BX information.
    sepRE = re.compile(r'[\]\[\s,;:]+')
    csv_input = open(args.inputFile, 'r')
    OldRun = -1

    LastValidLumi = []
    LastDelivered = 0

    output_line = '{'
    for line in csv_input:
        runLumiDict = {}    
        csvDict = {}

        if line[0] == '#':
            continue

        pieces = sepRE.split (line.strip())

        ipiece = 0

        if len(pieces) < 15:
            # The most likely cause of this is that we're using a csv file without the bunch data, so might as well
            # just give up now.
            raise RuntimeError("Not enough fields in input line; maybe you forgot to include --xing in your brilcalc command?\n"+line)
        try:
            run,       lumi     = int  ( pieces[0] ), int  ( pieces[2] )
            delivered, recorded = float( pieces[11] ), float( pieces[12] )
            luminometer = str( pieces[14] )
            xingInstLumiArray = [( int(orbit), float(lum), float(lumdelv) ) \
                                 for orbit, lum, lumdelv in zip( pieces[15::3],
                                                                 pieces[16::3],
                                                                 pieces[17::3]) ]
        except:
            print("Failed to parse line: check if the input format has changed")
            print(pieces[0],pieces[1],pieces[2],pieces[3],pieces[4],pieces[5],pieces[6],pieces[7],pieces[8],pieces[9])
            continue

        csvDict.setdefault (run, {})[lumi] = \
                           ( delivered, recorded, xingInstLumiArray )

        if run != OldRun:
            if OldRun>0:
                # add one empty lumi section at the end of each run, to mesh with JSON files
                LumiString = "[%d,0.0,0.0,0.0]," % (LastValidLumi[0]+1)
                output_line+=LumiString

                output_line = output_line[:-1] + '], '
            OldRun = run
            output_line+= ('"%d":' % run )
            output_line+= ' ['

            if lumi == 2:  # there is a missing LS=1 for this run
                output_line+= '[1,0.0,0.0,0.0],'

        for runNumber, lumiDict in sorted( six.iteritems(csvDict) ):
            LumiArray = CalcPileup(lumiDict, parameters, luminometer, selBX)

            LastValidLumi = LumiArray
            LastDelivered = lumiDict[LumiArray[0]][0] 

            LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (LumiArray[0], LumiArray[1], LumiArray[2], LumiArray[3])
            output_line += LumiString

    output_line = output_line[:-1] + ']}'
    csv_input.close()

    outputfile = open(output,'w')
    if not outputfile:
        raise RuntimeError("Could not open '%s' as an output JSON file" % output)

    outputfile.write(output_line)
    outputfile.close()
    print("Output written to", output)

    sys.exit()


