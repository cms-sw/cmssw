#!/usr/bin/env python
import os, sys
import coral
import array
import optparse
from RecoLuminosity.LumiDB import csvSelectionParser, selectionParser
import RecoLuminosity.LumiDB.lumiQueryAPI as LumiQueryAPI
import re

from pprint import pprint

def fillPileupHistogram (deadTable, parameters,
                         runNumber = 0, hist = None, debug = False,
                         mode='deadtable'):
    '''Given a deadtable and run number, will (create if necessary
    and) fill histogram with expected pileup distribution.  If a
    histogram is created, it is owned by the user and is his/her
    responsibility to clean up the memory.'''
    if hist:
        maxBin = hist.GetNbinsX()
        upper = int( hist.GetBinLowEdge(maxBin) + \
                     hist.GetBinWidth(maxBin) + 0.25 )
    else:
        histname = '%s_%s' % (parameters.pileupHistName, runNumber)
        hist = ROOT.TH1D (histname, histname, parameters.maxPileupBin + 1,
                          -0.5, parameters.maxPileupBin + 0.5)
        upper  = parameters.maxPileupBin
    for lumiSection, deadArray in sorted (deadTable.iteritems()):
        if mode == 'csv':
            numerator     = float (deadArray[1])
            denominator   = float (deadArray[0])
            instLumiArray =        deadArray[2]
            livetime = 1
            if numerator < 0:
                numerator = 0
            if denominator:
                livetime = numerator / denominator
        else:
            # we got everything from lumiDB
            if len(deadArray) <= parameters.xingIndex:
                # for some reason the xing instantaneous luminosity
                # information isn't there.  Print a warning and then skip
                # it:
                if parameters.noWarnings:
                    continue
                if runNumber:
                    print "No Xing Instantaneous luminosity information for run %d, lumi section %d" \
                          % (runNumber, lumiSection)
                else:
                    print "No Xing Instantaneous luminosity information for lumi section %d" \
                          % lumiSection
                continue
            numerator   = float (deadArray[0])
            denominator = float (deadArray[2] * deadArray[4])
            xingInstLumiArray = deadArray[parameters.xingIndex]
            # here we only want the instantaneous luminosities and don't
            # care which crosings they fall in.  So we only want the odd 
            instLumiArray = [(xingInstLumiArray[index], xingInstLumiArray[index + 1]) \
                             for index in xrange( 0, len (xingInstLumiArray), 2 ) ]
            livetime = 1
            if numerator < 0:
                numerator = 0
            if denominator:
                livetime = 1 - numerator / denominator
        # totalInstLumi = reduce(lambda x, y: x+y, instLumiArray) # not needed
        for xing, xingInstLumi in instLumiArray:
            xingIntLumi = xingInstLumi * parameters.lumiSectionLen * livetime
            mean = xingInstLumi * parameters.minBiasXsec * \
                   parameters.rotationTime
            if mean > 100:
                if runNumber:
                    print "mean number of pileup events > 100 for run %d, lum %d : m %f l %f" % \
                          (runNumber, lumiSection, mean, xingInstLumi)
                else:
                    print "mean number of pileup events > 100 for lum %d: m %f l %f" % \
                          (lumiSection, mean, xingInstLumi)
            totalProb = 0
            for obs in range (upper):
                prob = ROOT.TMath.Poisson (obs, mean)
                totalProb += prob
                hist.Fill (obs, prob * xingIntLumi)
            if debug:
                print "ls", lumiSection, "xing", xing, "inst", xingInstLumi, \
                      "mean", mean, "totalProb", totalProb, 1 - totalProb
                print "  hist mean", hist.GetMean()
            if totalProb < 1:
                hist.Fill (obs, (1 - totalProb) * xingIntLumi)
    return hist
    


##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == '__main__':
    parameters = LumiQueryAPI.ParametersObject()
    beamModeChoices = ["","stable", "quiet", "either"]
    parser = optparse.OptionParser ("Usage: %prog [--options] output.root",
                                    description = "Script to estimate pileup distribution using xing instantaneous luminosity information and minimum bias cross section.  Output is TH1D stored in root file")
    dbGroup     = optparse.OptionGroup (parser, "Database Options")
    inputGroup  = optparse.OptionGroup (parser, "Input Options")
    pileupGroup = optparse.OptionGroup (parser, "Pileup Options")
    dbGroup.add_option     ('--parameters', dest = 'connect', action = 'store',
                           help = 'connect string to lumiDB (optional, default to frontier://LumiCalc/CMS_LUMI_PROD)')
    dbGroup.add_option     ('-P', dest = 'authpath', action = 'store',
                             help = 'path to authentication file')
    dbGroup.add_option     ('--siteconfpath', dest = 'siteconfpath', action = 'store',
                             help = 'specific path to site-local-config.xml file, default to $CMS_PATH/SITECONF/local/JobConfig, ' \
                             'if path undefined, fallback to cern proxy&server')
    dbGroup.add_option     ('--debug', dest = 'debug', action = 'store_true',
                            help = 'debug')
    inputGroup.add_option  ('-r', dest = 'runnumber', action = 'store',
                            help = 'run number')
    inputGroup.add_option  ('-i', dest = 'inputfile', action = 'store',
                            help = 'lumi range selection file')
    inputGroup.add_option  ('-b', dest = 'beammode', default='', choices=beamModeChoices,
                            help = "beam mode, optional for delivered action, default ('%%default' out of %s)" % beamModeChoices)
    inputGroup.add_option  ('--lumiversion', dest = 'lumiversion', type='string', default='0001',
                            help = 'lumi data version, optional for all, default %default')
    inputGroup.add_option  ('--hltpath', dest = 'hltpath', action = 'store',
                           help = 'specific hltpath to calculate the recorded luminosity, default to all')
    inputGroup.add_option  ('--csvInput', dest = 'csvInput', type='string', default='',
                            help = 'Use CSV file from lumiCalc.py instead of lumiDB')
    pileupGroup.add_option ('--xingMinLum', dest = 'xingMinLum', type='float', default = 1e-3,
                            help = 'Minimum luminosity considered for "lsbylsXing" action, default %default')
    pileupGroup.add_option ('--minBiasXsec', dest = 'minBiasXsec', type='float', default = parameters.minBiasXsec,
                            help = 'Minimum bias cross section assumed (in mb), default %default mb')
    pileupGroup.add_option ('--histName', dest='histName', type='string', default = parameters.pileupHistName,
                            help = 'Histrogram name (default %default)')
    pileupGroup.add_option ('--maxPileupBin', dest='maxPileupBin', type='int', default = parameters.maxPileupBin,
                            help = 'Maximum bin in pileup histogram (default %default)')
    pileupGroup.add_option ('--saveRuns', dest='saveRuns', action='store_true',
                            help = 'Save pileup histograms for individual runs')
    pileupGroup.add_option ('--debugLumi', dest='debugLumi',
                            action='store_true',
                            help = 'Print out debug info for individual lumis')
    pileupGroup.add_option ('--nowarning', dest = 'nowarning',
                            action = 'store_true', default = False,
                            help = 'suppress bad for lumi warnings' )
    parser.add_option_group (dbGroup)
    parser.add_option_group (inputGroup)
    parser.add_option_group (pileupGroup)
    # parse arguments
    (options, args) = parser.parse_args()
    import ROOT 
    if not args:
        parser.print_usage()
        sys.exit()
    if len (args) != 1:
        parser.print_usage()
        raise RuntimeError, "Exactly one output file must be given"
    output = args[0]

    # get database session hooked up
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath

    ## Save what we need in the parameters object
    parameters.verbose        = True
    parameters.noWarnings     = options.nowarning
    parameters.lumiXing       = True
    parameters.lumiversion    = options.lumiversion
    if options.beammode=='stable':
        parameters.beammode    = 'STABLE BEAMS'
    parameters.xingMinLum     = options.xingMinLum
    parameters.minBiasXsec    = options.minBiasXsec
    parameters.pileupHistName = options.histName
    parameters.maxPileupBin   = options.maxPileupBin

    session, svc =  \
             LumiQueryAPI.setupSession (options.connect or \
                                        'frontier://LumiCalc/CMS_LUMI_PROD',
                                        options.siteconfpath, parameters,options.debug)

    ## Let's start the fun
    if not options.inputfile and not options.runnumber and not options.csvInput:
        raise "must specify either a run (-r), an input run selection file (-i), or an input CSV file (--csvInput)"
    pileupHist = ROOT.TH1D(parameters.pileupHistName, parameters.pileupHistName,
                      parameters.maxPileupBin + 1,
                      -0.5, parameters.maxPileupBin + 0.5)
    histList = []
    if options.csvInput:
        # we're going to read in the CSV file and use this as not only
        # the selection of which run/events to use, but also the
        # source of the lumi information.
        sepRE = re.compile (r'[\s,;:]+')
        runLumiDict = {}    
        events = open (options.csvInput, 'r')
        csvDict = {}
        for line in events:
            pieces = sepRE.split (line.strip())
            if len (pieces) < 6:
                continue
            if len (pieces) % 2:
                # not an even number
                continue
            try:
                run,       lumi     = int  ( pieces[0] ), int  ( pieces[1] )
                delivered, recorded = float( pieces[2] ), float( pieces[3] )
                xingInstLumiArray = [( int(orbit), float(lum) ) \
                                     for orbit, lum in zip( pieces[4::2],
                                                            pieces[5::2] ) ]
            except:
                continue
            csvDict.setdefault (run, {})[lumi] = \
                               ( delivered, recorded, xingInstLumiArray )
        events.close()
        for runNumber, lumiDict in sorted( csvDict.iteritems() ):
            if options.saveRuns:
                hist = fillPileupHistogram (lumiDict, parameters,
                                            runNumber = runNumber,
                                            debug = options.debugLumi,
                                            mode='csv')
                pileupHist.Add (hist)
                histList.append (hist)
            else:
                fillPileupHistogram (lumiDict, parameters,
                                     hist = pileupHist,
                                     debug = options.debugLumi,
                                     mode='csv')
            
        histFile = ROOT.TFile.Open (output, 'recreate')
        if not histFile:
            raise RuntimeError, \
                  "Could not open '%s' as an output root file" % output
        pileupHist.Write()
        for hist in histList:
            hist.Write()
        histFile.Close()
        #pprint (csvDict)
        sys.exit()

        
    ## Get input source
    if options.runnumber:
        inputRange = options.runnumber
    else:
        basename, extension = os.path.splitext (options.inputfile)
        if extension == '.csv': # if file ends with .csv, use csv
                                # parser, else parse as json file
            fileparsingResult = csvSelectionParser.csvSelectionParser (options.inputfile)
        else:
            f = open (options.inputfile, 'r')
            inputfilecontent = f.read()
            inputRange =  selectionParser.selectionParser (inputfilecontent)
        if not inputRange:
            print 'failed to parse the input file', options.inputfile
            raise 

    recordedData  = LumiQueryAPI.recordedLumiForRange  (session, parameters, inputRange)
    ## pprint (recordedData)
    for runDTarray in recordedData:
        runNumber = runDTarray[0]
        deadTable = runDTarray[2]
        if options.saveRuns:
            hist = fillPileupHistogram (deadTable, parameters,
                                        runNumber = runNumber,
                                        debug = options.debugLumi)
            pileupHist.Add (hist)
            histList.append (hist)
        else:
            fillPileupHistogram (deadTable, parameters,
                                 hist = pileupHist,
                                 debug = options.debugLumi)
    histFile = ROOT.TFile.Open (output, 'recreate')
    if not histFile:
        raise RuntimeError, \
              "Could not open '%s' as an output root file" % output
    pileupHist.Write()
    for hist in histList:
        hist.Write()
    histFile.Close()
        
