#!/usr/bin/env python
from __future__ import print_function
from builtins import range
VERSION='2.00'
import os, sys
import coral
import optparse
from RecoLuminosity.LumiDB import sessionManager,csvSelectionParser,selectionParser,lumiCorrections,lumiCalcAPI
import six

beamChoices=['PROTPHYS','IONPHYS']

class pileupParameters(object):
    def __init__(self):
        self.minBiasXsec=71300 #unit microbarn
        self.rotationRate=11245.613 # for 3.5 TeV Beam energy
        self.NBX=3564
        self.rotationTime=1.0/self.rotationRate
        self.lumiSectionLen=2**18*self.rotationTime

def fillPileupHistogram (bxlumiinfo,pileupHistName,maxPileupBin,
                         runNumber=0, hist = None, debug = False):
    '''
    bxlumiinfo:[[cmslsnum(0),avgdelivered(1),avgrecorded(2),bxlumiarray[3]]]

    Given luminfo , deadfrac info and run number, will (create if necessary
    and) fill histogram with expected pileup distribution.  If a
    histogram is created, it is owned by the user and is his/her
    responsibility to clean up the memory.'''
    if hist:
        maxBin = hist.GetNbinsX()
        upper = int( hist.GetBinLowEdge(maxBin) + \
                     hist.GetBinWidth(maxBin) + 0.25 )
    else:
        histname = '%s_%s' % (pileupHistName, runNumber)
        hist = ROOT.TH1D (histname, histname, maxPileupBin + 1,
                          -0.5,maxPileupBin + 0.5)
        upper = maxPileupBin
    p=pileupParameters()
    for perlsinfo in bxlumiinfo:
        cmslsnum=perlsinfo[0]
        avgdelivered=perlsinfo[1]
        avgrecorded=perlsinfo[2]
        bxdata=perlsinfo[3]
        bxidx=bxdata[0]
        bxvaluelist=bxdata[1]
        #calculate livefrac
        livetime=1
        if avgrecorded<0:
            avgrecorded=0
        if avgdelivered:
            livetime=avgrecorded/avgdelivered
        else:
            livetime=0
        for idx,bxvalue in enumerate(bxvaluelist):
            xingIntLumi=bxvalue * p.lumiSectionLen * livetime
            if options.minBiasXsec:
                mean = bxvalue * options.minBiasXsec * p.rotationTime
            else:
                mean = bxvalue * p.minBiasXsec * p.rotationTime
            if mean > 100:
                if runNumber:
                    print("mean number of pileup events > 100 for run %d, lum %d : m %f l %f" % \
                          (runNumber, lumiSection, mean, bxvalue))
                else:
                    print("mean number of pileup events > 100 for lum %d: m %f l %f" % \
                          (cmslsnum, mean, bxvalue))
            totalProb = 0
            for obs in range (upper):
                prob = ROOT.TMath.Poisson (obs, mean)
                totalProb += prob
                hist.Fill (obs, prob * xingIntLumi)
            if debug:
                xing=bxidx[idx]
                print("ls", lumiSection, "xing", xing, "inst", bxvalue, \
                      "mean", mean, "totalProb", totalProb, 1 - totalProb)
                print("  hist mean", hist.GetMean())
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
    beamModeChoices = [ "","stable", "quiet", "either"]
    amodetagChoices = [ "PROTPHYS","IONPHYS" ]
    xingAlgoChoices =[ "OCC1","OCC2","ET"]
    parser = optparse.OptionParser ("Usage: %prog [--options] output.root",
                                    description = "Script to estimate pileup distribution using xing instantaneous luminosity information and minimum bias cross section.  Output is TH1D stored in root file")
    dbGroup     = optparse.OptionGroup (parser, "Database Options")
    inputGroup  = optparse.OptionGroup (parser, "Input Options")
    pileupGroup = optparse.OptionGroup (parser, "Pileup Options")
    dbGroup.add_option     ('-c', dest = 'connect', action = 'store',
                            default='frontier://LumiCalc/CMS_LUMI_PROD',
                            help = 'connect string to lumiDB ,default %default')
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
    inputGroup.add_option  ('-b', dest = 'beamstatus', default='', choices=beamModeChoices,
                            help = "beam mode, optional for delivered action, default ('%%default' out of %s)" % beamModeChoices)
    inputGroup.add_option  ('--hltpath', dest = 'hltpath', action = 'store',
                            help = 'specific hltpath to calculate the recorded luminosity, default to all')
    inputGroup.add_option  ('--csvInput', dest = 'csvInput', type='string', default='',
                            help = 'Use CSV file from lumiCalc2.py instead of lumiDB')
    inputGroup.add_option  ('--without-correction', dest = 'withoutFineCorrection', action='store_true',
                            help = 'not apply fine correction on calibration')
    pileupGroup.add_option('--algoname',dest='algoname',default='OCC1',
                            help = 'lumi algorithm , default %default')
    pileupGroup.add_option ('--xingMinLum', dest = 'xingMinLum', type='float', default = 1.0e-3,
                            help = 'Minimum luminosity considered for "lsbylsXing" action, default %default')
    pileupGroup.add_option ('--minBiasXsec', dest = 'minBiasXsec', type='float', default = 71300,
                            help = 'Minimum bias cross section assumed (in mb), default %default mb')
    pileupGroup.add_option ('--histName', dest='pileupHistName', type='string', default = 'pileup',
                            help = 'Histrogram name (default %default)')
    pileupGroup.add_option ('--maxPileupBin', dest='maxPileupBin', type='int', default = 10,
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
    try:
        (options, args) = parser.parse_args()
    except Exception as e:
        print(e)
    if not args:
        parser.print_usage()
        sys.exit()
    if len (args) != 1:
        parser.print_usage()
        raise RuntimeError("Exactly one output file must be given")
    output = args[0]
    finecorrections=None
    # get database session hooked up
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath

    svc=sessionManager.sessionManager(options.connect,authpath=options.authpath,debugON=options.debug)

    if not options.inputfile and not options.runnumber and not options.csvInput:
        raise "must specify either a run (-r), an input run selection file (-i), or an input CSV file (--csvInput)"

    runDict = {}
    if options.csvInput:
        import re
        # we're going to read in the CSV file and use this as not only
        # the selection of which run/events to use, but also the
        # source of the lumi information.
        sepRE = re.compile (r'[\s,;:]+')
        events = open (options.csvInput, 'r')
        for line in events:
            pieces = sepRE.split (line.strip())
            if len (pieces) < 6:
                continue
            if len (pieces) % 2:
                # not an even number
                continue
            try:
                run,       cmslsnum = int  ( pieces[0] ), int  ( pieces[1] )
                delivered, recorded = float( pieces[2] ), float( pieces[3] )
                xingIdx = [int(myidx) for myidx in  pieces[4::2] ]
                xingVal = [float(myval) for myval in pieces[5::2] ]
                
                #xingInstLumiArray = [( int(orbit), float(lum) ) \
                #                     for orbit, lum in zip( pieces[4::2],
                #                                            pieces[5::2] ) ]
            except:
                continue
            runDict.setdefault(run,[]).append([cmslsnum,delivered,recorded,(xingIdx,xingVal)])
            #{run:[[cmslsnum,delivered,recorded,xingInstlumiArray]..]}
        events.close()
    else:
        session=svc.openSession(isReadOnly=True,cpp2sqltype=[('unsigned int','NUMBER(10)'),('unsigned long long','NUMBER(20)')])
        finecorrections=None
        if options.runnumber:
            inputRange = {int(options.runnumber):None}
        else:
            basename, extension = os.path.splitext (options.inputfile)
            if extension == '.csv': # if file ends with .csv, use csv
                # parser, else parse as json file
                fileparsingResult = csvSelectionParser.csvSelectionParser (options.inputfile)
            else:
                f = open (options.inputfile, 'r')
                inputfilecontent = f.read()
                inputRange =  selectionParser.selectionParser (inputfilecontent).runsandls()
        if not inputRange:
            print('failed to parse the input file', options.inputfile)
            raise
        if not options.withoutFineCorrection:
            rruns=inputRange.keys()
            schema=session.nominalSchema()
            session.transaction().start(True)
            finecorrections=lumiCorrections.correctionsForRange(schema,rruns)
            session.transaction().commit()
        session.transaction().start(True)
        schema=session.nominalSchema()
        lumiData=lumiCalcAPI.lumiForRange(schema,inputRange,beamstatus=options.beamstatus,withBXInfo=True,bxAlgo=options.algoname,xingMinLum=options.xingMinLum,withBeamIntensity=False,datatag=None,finecorrections=finecorrections)
        session.transaction().commit()
        # {run:[lumilsnum(0),cmslsnum(1),timestamp(2),beamstatus(3),beamenergy(4),deliveredlumi(5),recordedlumi(6),calibratedlumierror(7),(bxidx,bxvalues,bxerrs)(8),None]}}
    ##convert lumiData to lumiDict format #{run:[[cmslsnum,avg]]}
        for runnum,perrundata in lumiData.items():
            bxlumiinfo=[]
            for perlsdata in perrundata:
                cmslsnum=perlsdata[1]
                deliveredlumi=perlsdata[5]
                recordedlumi=perlsdata[6]
                bxlist=perlsdata[8]
                bxlumiinfo.append([cmslsnum,deliveredlumi,recordedlumi,bxlist])
                runDict.setdefault(runnum,[]).append([cmslsnum,deliveredlumi,recordedlumi,bxlist])
    #print 'runDict ',runDict
    
    import ROOT 
    pileupHist = ROOT.TH1D (options.pileupHistName,options.pileupHistName,
                      options.maxPileupBin + 1,
                      -0.5, options.maxPileupBin + 0.5)
    histList = []
    for runNumber, lumiList in sorted( six.iteritems(runDict) ):
        if options.saveRuns:
            hist = fillPileupHistogram (lumiList,options.pileupHistName,options.maxPileupBin,
                                        runNumber = runNumber,
                                        debug = options.debugLumi)
            pileupHist.Add (hist)
            histList.append (hist)
        else:
            fillPileupHistogram (lumiList, options.pileupHistName,options.maxPileupBin,
                                 hist = pileupHist,
                                 debug = options.debugLumi)            
    histFile = ROOT.TFile.Open (output, 'recreate')
    if not histFile:
        raise RuntimeError("Could not open '%s' as an output root file" % output)
    pileupHist.Write()
    for hist in histList:
        hist.Write()
    histFile.Close()
    sys.exit()
