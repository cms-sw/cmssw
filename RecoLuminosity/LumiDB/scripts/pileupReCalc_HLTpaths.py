#!/usr/bin/env python
from __future__ import print_function
VERSION='1.00'
import os,sys,time
import optparse
from RecoLuminosity.LumiDB import pileupParser
from RecoLuminosity.LumiDB import selectionParser
from RecoLuminosity.LumiDB import csvLumibyLSParser
from math import exp
from math import sqrt
import six

def parseInputFile(inputfilename):
    '''
    output ({run:[ls:[inlumi, meanint]]})
    '''
    selectf=open(inputfilename,'r')
    inputfilecontent=selectf.read()
    p=pileupParser.pileupParser(inputfilecontent)                            
    
#    p=inputFilesetParser.inputFilesetParser(inputfilename)
    runlsbyfile=p.runsandls()
    return runlsbyfile



##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == '__main__':

    parser = optparse.OptionParser ("Usage: %prog [--options]",
                                    description = "Script to rescale pileup distributions using inputs derived by calculating luminosity for a given set of HLT paths.  Input format must be -lumibyls-")
#
#    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Pileup Lumi Calculation",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    CalculationModeChoices = ['truth', 'observed']

    #
    # parse arguments
    #  
    #
    # basic arguments
    #
    #parser.add_argument('action',choices=allowedActions,
    #                    help='command actions')
    parser.add_option('-o',dest='outputfile',action='store',
                        default='PileupRecalcJSON.txt',
                        help='output pileup JSON file')
    parser.add_option('-i',dest='inputfile',action='store',
                        help='Input Run/LS/lumis file for your trigger selection  (required)')
    parser.add_option('--inputLumiJSON',dest='inputLumiJSON',action='store',
                        help='Input Lumi/Pileup file in JSON format (required)')
    parser.add_option('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_option('--runperiod',dest='runperiod',action='store', default='Run1',help='select runperiod Run1 or Run2, default Run1' )
    # parse arguments
    try:
        (options, args) = parser.parse_args()
    except Exception as e:
        print(e)
#    if not args:
#        parser.print_usage()
#        sys.exit()
#    if len (args) != 1:
#        parser.print_usage()
#        raise RuntimeError, "Exactly one output file must be given"
#    output = args[0]
    
#    options=parser.parse_args()

    if options.verbose:
        print('General configuration')
        print('\toutputfile: ',options.outputfile)
        print('\tinput selection file: ',options.inputfile)

    #print options.runperiod
    #inpf = open (options.inputfile, 'r')
    #inputfilecontent = inpf.read()
      
    inputRange =  csvLumibyLSParser.csvLumibyLSParser (options.inputfile,options.runperiod).runsandls()

    #print 'number of runs processed %d' % csvLumibyLSParser.csvLumibyLSParser (options.inputfile).numruns()

    #inputRange=inputFilesetParser.inputFilesetParser(options.inputfile)

    
    inputPileupRange=parseInputFile(options.inputLumiJSON)

    # now, we have to find the information for the input runs and LumiSections 
    # in the Lumi/Pileup list. First, loop over inputs

    OUTPUTLINE = ""
    OUTPUTLINE+='{'

    for (run, lslist) in sorted (six.iteritems(inputRange)):
        # now, look for matching run, then match lumi sections
        #print "searching for run %d" % (run)
        if run in inputPileupRange.keys():
            OUTPUTLINE+= ('"%d":' % run )
            OUTPUTLINE+= ' ['
            
            LSPUlist = inputPileupRange[run]
            #print "LSPUlist", LSPUlist
            for LSnumber in lslist:
                if LSnumber in LSPUlist.keys():
                    PUlumiInfo = LSPUlist[LSnumber]
                    HLTlumiInfo = lslist[LSnumber]
                    #print "found LS %d" % (LSnumber)
                    #print HLTlumiInfo
                    scale = 0
                    if PUlumiInfo[0] > 0.:
                        scale=HLTlumiInfo[1]/PUlumiInfo[0] # rescale to HLT recorded Lumi

                    if scale > 1.001:
                        print('Run %d, LS %d, HLT Scale (%f), HLTL (%f), PUL (%f) larger than one - please check!' % (run, LSnumber, scale, HLTlumiInfo[1],PUlumiInfo[0]))
                        scale=1.01  # HLT integrated values are wrong, punt                        

                    newIntLumi = scale*PUlumiInfo[0]
                    newRmsLumi = PUlumiInfo[1]
                    newInstLumi = PUlumiInfo[2]
                    if scale == 0:
                        newInstLumi = PUlumiInfo[2]  # keep non-zero value, with zero weight
                                                     # to avoid spike at zero interactions
                    #print PUlumiInfo[0],HLTlumiInfo[1]
                    LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (LSnumber, newIntLumi, newRmsLumi ,newInstLumi)
                    OUTPUTLINE += LumiString

                    #for study
                    #print '%d %d %f %f' % (run,LSnumber,PUlumiInfo[0],HLTlumiInfo[1])

                else: # no match, use zero for int lumi
                    newInstLumi = 10.0  # keep non-zero value, with zero weight
                                        # to avoid spike at zero interactions
                    #print PUlumiInfo[0],HLTlumiInfo[1]
                    LumiString = "[%d,0.0,0.0,%2.4e]," % (LSnumber, newInstLumi)
                    OUTPUTLINE += LumiString
                    

            lastindex=len(OUTPUTLINE)-1
            trunc = OUTPUTLINE[0:lastindex]
            OUTPUTLINE = trunc
            OUTPUTLINE += '], '

        else:  # trouble
            print("Run %d not found in Lumi/Pileup input file.  Check your files!" % (run))


#            print run
#            print lslist

#        histFile = ROOT.TFile.Open (output, 'recreate')
#        if not histFile:
#            raise RuntimeError, \
#                 "Could not open '%s' as an output root file" % output
#        pileupHist.Write()
        #for hist in histList:
        #    hist.Write()
#        histFile.Close()
#        sys.exit()

    lastindex=len(OUTPUTLINE)-2
    trunc = OUTPUTLINE[0:lastindex]
    OUTPUTLINE = trunc
    OUTPUTLINE += ' }'

    outputfile = open(options.outputfile,'w')
    if not outputfile:
        raise RuntimeError("Could not open '%s' as an output JSON file" % output)
                    
    outputfile.write(OUTPUTLINE)
    outputfile.close()



    sys.exit()
