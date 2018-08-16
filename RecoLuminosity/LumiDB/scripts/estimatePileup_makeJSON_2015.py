#!/usr/bin/env python
from __future__ import print_function
import os, sys
import coral
import array
import optparse
from RecoLuminosity.LumiDB import csvSelectionParser, selectionParser
import RecoLuminosity.LumiDB.lumiQueryAPI as LumiQueryAPI
import re
from math import sqrt

from pprint import pprint
import six

selBXs=[]
def CalcPileup (deadTable, parameters, luminometer, mode='deadtable'):
    '''Given a deadtable, will calculate parameters of pileup distribution. Return formatted
    string with LumiSection, LS integrated lumi, RMS of bunch to bunch lumi and pileup.'''

    LumiString = ""
    LumiArray = []

    for lumiSection, deadArray in sorted (six.iteritems(deadTable)):
        numerator = 0
        if luminometer == "HFOC":
            threshold = 8.
        else:
            threshold = 1.2
        if mode == 'csv':
            numerator     = float (deadArray[1])
            denominator   = float (deadArray[0])
            instLumiArray =        deadArray[2]
            livetime = 1
            if numerator < 0:
                numerator = 0
            if denominator:
                livetime = numerator / denominator



            #print "params", parameters.lumiSectionLen, parameters.rotationTime

        else:
            print("no csv input! Doh!")
            return
        # totalInstLumi = reduce(lambda x, y: x+y, instLumiArray) # not needed
        if lumiSection > 0:
            TotalLumi = 0 
            TotalInt = 0
            TotalInt2 = 0
            TotalWeight = 0
            TotalWeight2 = 0
            FilledXings = 0
            for xing, xingInstLumi, xingDelvLumi in instLumiArray:
                if selBXs and xing not in selBXs:
                    continue
                xingIntLumi = xingInstLumi * livetime  # * parameters.lumiSectionLen
                mean = xingInstLumi * parameters.rotationTime / parameters.lumiSectionLen
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
                if selBXs and xing not in selBXs:
                    continue
                if xingInstLumi > threshold:
                    xingIntLumi = xingInstLumi * livetime # * parameters.lumiSectionLen
                    mean = xingInstLumi * parameters.rotationTime / parameters.lumiSectionLen
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
            lumiX=MeanInt*parameters.lumiSectionLen*FilledXings*(1./parameters.rotationTime)

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
# As of now, .csv file is the only allowed input.  See the old script for a variety
# of different ways to access the data.

if __name__ == '__main__':
    parameters = LumiQueryAPI.ParametersObject()
    parser = optparse.OptionParser ("Usage: %prog [--options] output.root",
                                    description = "Script to estimate average instantaneous bunch crossing luminosity using xing instantaneous luminosity information. Output is JSON format file with one entry per LumiSection")
    inputGroup  = optparse.OptionGroup (parser, "Input Options")
    pileupGroup = optparse.OptionGroup (parser, "Pileup Options")
    inputGroup.add_option  ('--csvInput', dest = 'csvInput', type='string', default='',
                            help = 'Use CSV file from lumiCalc.py instead of lumiDB')
    inputGroup.add_option  ('--selBXs', dest = 'selBXs', type='string', default='',
                            help = 'CSV of BXs to use; if empty, select all')
    parser.add_option_group (inputGroup)
    parser.add_option_group (pileupGroup)
    # parse arguments
    (options, args) = parser.parse_args()

    if not args:
        parser.print_usage()
        sys.exit()
    if len (args) != 1:
        parser.print_usage()
        raise RuntimeError("Please specify an output file as your last argument")
    output = args[0]

    ## Let's start the fun
    if not options.csvInput:
        raise "you must specify an input CSV file with (--csvInput)"

    if options.selBXs != "":
        for iBX in options.selBXs.split(","):
            try:
                BX=int(iBX)
                if BX not in selBXs:
                    selBXs.append(BX)
            except:
                print(iBX,"is not an int")
        selBXs.sort()
    print("selBXs",selBXs)

    OUTPUTLINE = ""
    if options.csvInput:
        # we're going to read in the CSV file and use this as not only
        # the selection of which run/events to use, but also the
        # source of the lumi information.
        sepRE = re.compile (r'[\]\[\s,;:]+')
        events = open (options.csvInput, 'r')
        OldRun = -1

        InGap = 0;
        GapDict = {}
        LastValidLumi = []
        LastDelivered = 0


        OUTPUTLINE+='{'
        for line in events:
            runLumiDict = {}    
            csvDict = {}

            if re.search( '^#', line):
                continue

            pieces = sepRE.split (line.strip())

            ipiece = 0

            if len (pieces) < 15: # means we are missing data; keep track of LS, lumi
                InGap = 1
                try:
                    run,       lumi     = int  ( pieces[0] ), int  ( pieces[2] )
                    delivered, recorded = float( pieces[11] ), float( pieces[12] )
                except:
                    if pieces[0] != 'run':
                        print(" cannot parse csv file ")
                    InGap = 0
                    continue
                GapDict[lumi] = [delivered, recorded]
                continue
            #if len (pieces) % 2:
                # not an even number
            #    continue
            try:
                run,       lumi     = int  ( pieces[0] ), int  ( pieces[2] )
                delivered, recorded = float( pieces[11] ), float( pieces[12] )
                luminometer = str( pieces[14] )
                xingInstLumiArray = [( int(orbit), float(lum), float(lumdelv) ) \
                                     for orbit, lum, lumdelv in zip( pieces[15::3],
                                                                     pieces[16::3],
                                                                     pieces[17::3]) ]
            except:
                print(" Bad Parsing: Check if the input format has changed")
                print(pieces[0],pieces[1],pieces[2],pieces[3],pieces[4],pieces[5],pieces[6],pieces[7],pieces[8],pieces[9])
                continue

            csvDict.setdefault (run, {})[lumi] = \
                               ( delivered, recorded, xingInstLumiArray )#( delivered, recorded, xingInstLumiArray )

            if run != OldRun:
                if OldRun>0:
                    if InGap == 1:  # We have some LS's at the end with no data
                        lastLumiS = 0
                        for lumiS, lumiInfo in sorted ( six.iteritems(GapDict) ):
                            record = lumiInfo[1]
                            lastLumiS = lumiS
                            if record > 0.01:

                                peakratio = lumiInfo[0]/LastDelivered # roughly, ratio of inst lumi
                                pileup = LastValidLumi[3]*peakratio     # scale for this LS
                                aveLumi = 0
                                if lumiInfo[0] >0:
                                    aveLumi = LastValidLumi[1]*peakratio*lumiInfo[1]/lumiInfo[0]  # scale by rec/del
                                LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (lumiS, aveLumi, LastValidLumi[2],pileup)
                                OUTPUTLINE += LumiString
                            else:
                                LumiString = "[%d,0.0,0.0,0.0]," % (lumiS)
                                OUTPUTLINE+=LumiString
                        #add one empty one at the end
                        LumiString = "[%d,0.0,0.0,0.0]," % (lastLumiS+1)
                        OUTPUTLINE+=LumiString

                        InGap = 0
                        GapDict.clear()

                    # add one empty lumi section at the end of each run, to mesh with JSON files
                    LumiString = "[%d,0.0,0.0,0.0]," % (LastValidLumi[0]+1)
                    OUTPUTLINE+=LumiString

                    lastindex=len(OUTPUTLINE)-1
                    trunc = OUTPUTLINE[0:lastindex]
                    OUTPUTLINE = trunc
                    OUTPUTLINE += '], '
                OldRun = run
                OUTPUTLINE+= ('"%d":' % run )
                OUTPUTLINE+= ' ['

                if lumi == 2:  # there is a missing LS=1 for this run
                    OUTPUTLINE+= '[1,0.0,0.0,0.0],'

            for runNumber, lumiDict in sorted( six.iteritems(csvDict) ):
        #	print runNumber
                LumiArray = CalcPileup (lumiDict, parameters, luminometer,
                                     mode='csv')

                LastValidLumi = LumiArray
                LastDelivered = lumiDict[LumiArray[0]][0] 

                if InGap == 1:  # We have some gap before this in entry in this run
                    for lumiS, lumiInfo in sorted ( six.iteritems(GapDict) ):
                        peakratio = lumiInfo[0]/LastDelivered # roughly, ratio of inst lumi
                        pileup = LumiArray[3]*peakratio     # scale for this LS
                        aveLumi = 0
                        if lumiInfo[0] > 0:
                            aveLumi = LumiArray[1]*peakratio*lumiInfo[1]/lumiInfo[0]  # scale by rec/del
                        LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (lumiS, aveLumi, LumiArray[2],pileup)
                        OUTPUTLINE += LumiString
                    InGap = 0
                LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (LumiArray[0], LumiArray[1], LumiArray[2], LumiArray[3])
                OUTPUTLINE += LumiString


        lastindex=len(OUTPUTLINE)-1
        trunc = OUTPUTLINE[0:lastindex]
        OUTPUTLINE = trunc
        OUTPUTLINE += ']}'
        events.close()

        outputfile = open(output,'w')
        if not outputfile:
            raise RuntimeError("Could not open '%s' as an output JSON file" % output)

        outputfile.write(OUTPUTLINE)
        outputfile.close()

        sys.exit()


