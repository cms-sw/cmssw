#!/usr/bin/env python
VERSION='1.00'
import os,sys,time
import optparse
from RecoLuminosity.LumiDB import pileupParser
from RecoLuminosity.LumiDB import selectionParser
from RecoLuminosity.LumiDB import csvLumibyLSParser
from math import exp
from math import sqrt

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
                                    description = "Script to rescale pileup distributions using inputs derived by calculating luminosity for a given set of external corrections (Pixel luminosity, for example).  Input format must be -lumibyls-")
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
    
    # parse arguments
    try:
        (options, args) = parser.parse_args()
    except Exception as e:
        print e
#    if not args:
#        parser.print_usage()
#        sys.exit()
#    if len (args) != 1:
#        parser.print_usage()
#        raise RuntimeError, "Exactly one output file must be given"
#    output = args[0]
    
#    options=parser.parse_args()

    if options.verbose:
        print 'General configuration'
        print '\toutputfile: ',options.outputfile
        print '\tinput selection file: ',options.inputfile


    #inpf = open (options.inputfile, 'r')
    #inputfilecontent = inpf.read()
    inputRange =  csvLumibyLSParser.csvLumibyLSParser (options.inputfile).runsandls()

    #print 'number of runs processed %d' % csvLumibyLSParser.csvLumibyLSParser (options.inputfile).numruns()

    #inputRange=inputFilesetParser.inputFilesetParser(options.inputfile)

    
    inputPileupRange=parseInputFile(options.inputLumiJSON)

    # now, we have to find the information for the input runs and LumiSections 
    # in the Lumi/Pileup list. First, loop over inputs

    OUTPUTLINE = ""
    OUTPUTLINE+='{'

    # loop over pileup JSON as source, since it should have more lumi sections

    for (run, LSPUlist) in sorted (inputPileupRange.iteritems() ):
        # now, look for matching run, then match lumi sections
        #print "searching for run %d" % (run)
        if run in inputRange.keys():
            OUTPUTLINE+= ('"%d":' % run )
            OUTPUTLINE+= ' ['
            
            lslist = inputRange[run]
            #print "LSPUlist", LSPUlist
            for LSnumber in LSPUlist:
                if LSnumber in lslist.keys():  # do we find a match in pixel list for this LS?
                    PUlumiInfo = LSPUlist[LSnumber]
                    PixlumiInfo = lslist[LSnumber]
                    #print "found LS %d" % (LSnumber)
                    #print HLTlumiInfo
                    scale = 0
                    if PUlumiInfo[0] > 0.:
                        scale=PixlumiInfo[1]/PUlumiInfo[0] # rescale to HLT recorded Lumi

                    if scale !=0 and (scale < 0.2 or scale > 5.0):
                        print 'Run %d, LS %d, Scale (%f), PixL (%f), PUL (%f) big change - please check!' % (run, LSnumber, scale, PixlumiInfo[1],PUlumiInfo[0])
                    #    scale=1.01  # HLT integrated values are wrong, punt                        

                    newIntLumi = scale*PUlumiInfo[0]
                    newRmsLumi = scale*PUlumiInfo[1]
                    newInstLumi = scale*PUlumiInfo[2]
                    if scale == 0:   # keep old HF values - maybe lumis was zero anyway
                        newIntLumi = PUlumiInfo[0]
                        newRmsLumi = PUlumiInfo[1]
                        newInstLumi = PUlumiInfo[2]
                                                     
                        print 'Run %d, LS %d, Scale (%f), PixL (%f), PUL (%f) - 0 please check!' % (run, LSnumber, scale, PixlumiInfo[1],PUlumiInfo[0])
                    LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (LSnumber, newIntLumi, newRmsLumi ,newInstLumi)
                    OUTPUTLINE += LumiString

                    #for study
                    #print '%d %d %f %f' % (run,LSnumber,PUlumiInfo[0],HLTlumiInfo[1])

                else: # no match, keep HF values
                    newIntLumi = PUlumiInfo[0]
                    newRmsLumi = PUlumiInfo[1]
                    newInstLumi = PUlumiInfo[2]

                    #print PUlumiInfo[0],HLTlumiInfo[1]
                    LumiString = "[%d,%2.4e,%2.4e,%2.4e]," % (LSnumber, newIntLumi, newRmsLumi ,newInstLumi)
                    OUTPUTLINE += LumiString
                    

            lastindex=len(OUTPUTLINE)-1
            trunc = OUTPUTLINE[0:lastindex]
            OUTPUTLINE = trunc
            OUTPUTLINE += '], '

        else:  # trouble
            print "Run %d not found in Lumi/Pileup input file.  Check your files!" % (run)


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
