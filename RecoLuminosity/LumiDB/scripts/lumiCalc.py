#!/usr/bin/env python
VERSION='2.00'
import os,sys
import coral
#import optparse
from RecoLuminosity.LumiDB import csvSelectionParser, selectionParser,argparse
import RecoLuminosity.LumiDB.lumiQueryAPI as LumiQueryAPI
#from pprint import pprint

##############################
## ######################## ##
## ## ################## ## ##
## ## ## Main Program ## ## ##
## ## ################## ## ##
## ######################## ##
##############################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),description = "Lumi Calculations",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    allowedActions = ['overview', 'delivered', 'recorded', 'lumibyls', 'lumibylsXing']
    beamModeChoices = [ "stable", "quiet", "either"]
    # parse arguments
    parser.add_argument('action',choices=allowedActions,help='command actions')
    parser.add_argument('-c',dest='connect',action='store',required=False,help='connect string to lumiDB,optional',default='frontier://LumiProd/CMS_LUMI_PROD')
    parser.add_argument('-P',dest='authpath',action='store',help='path to authentication file,optional')
    parser.add_argument('-n',dest='normfactor',action='store',type=float,default=1.0,help='normalization factor,optional')
    parser.add_argument('-r',dest='runnumber',action='store',type=int,help='run number,optional')
    parser.add_argument('-i',dest='inputfile',action='store',help='lumi range selection file,optional')
    parser.add_argument('-o',dest='outputfile',action='store',help='output to csv file,optional')
    parser.add_argument('-b',dest='beammode',action='store',choices=beamModeChoices,help='beam mode choice',default='stable')
    parser.add_argument('-lumiversion',dest='lumiversion',action='store',default='0001',help='lumi data version, optional')
    parser.add_argument('-hltpath',dest='hltpath',action='store',default='all',help='specific hltpath to calculate the recorded luminosity,optional')
    parser.add_argument('-siteconfpath',dest='siteconfpath',action='store',help='specific path to site-local-config.xml file, optional. If path undefined, fallback to cern proxy&server')
    parser.add_argument('-xingMinLum', dest = 'xingMinLum', type='float', default=1e-3,required=False,help='Minimum luminosity considered for "lsbylsXing" action')
    parser.add_argument('--verbose',dest='verbose',action='store_true',help='verbose mode for printing' )
    parser.add_argument('--nowarning',dest='nowarning',action='store_true',help='suppress bad for lumi warnings' )
    parser.add_argument('--debug',dest='debug',action='store_true',help='debug')
    options=parser.parse_args()
    if options.authpath:
        os.environ['CORAL_AUTH_PATH'] = options.authpath
    ## Let's start the fun
    if not options.inputfile and not options.runnumber:
        print "must specify either a run (-r) or an input run selection file (-i)"
        sys.exit()
        
    session,svc =  LumiQueryAPI.setupSession (options.connect or \
                                              'frontier://LumiProd/CMS_LUMI_PROD',
                                               options.siteconfpath, options.debug)

    ## Save what we need in the parameters object
    parameters = LumiQueryAPI.ParametersObject()
    parameters.verbose     = options.verbose
    parameters.noWarnings  = options.nowarning
    parameters.norm        = options.normfactor
    parameters.lumiversion = options.lumiversion
    parameters.beammode    = options.beammode
    parameters.xingMinLum  = options.xingMinLum
    
    lumiXing = False
    if options.action == 'lumibylsXing':
        #action = 'lumibyls'
        parameters.lumiXing = True
        # we can't have lumiXing mode if we're not writing to a CSV
        # file
        #if not options.outputfile:
        #    raise RuntimeError, "You must specify an outputt file in 'lumibylsXing' mode"
    if options.runnumber:
        inputRange = str(options.runnumber)
    else:
        basename, extension = os.path.splitext (options.inputfile)
        if extension == '.csv': # if file ends with .csv, use csv parser, else parse as json file
            fileparsingResult = csvSelectionParser.csvSelectionParser (options.inputfile)
        else:
            f = open (options.inputfile, 'r')
            inputfilecontent = f.read()
            inputRange =  selectionParser.selectionParser (inputfilecontent)
        if not inputRange:
            print 'failed to parse the input file', options.inputfile
            raise 

    # Delivered
    if options.action ==  'delivered':
        lumidata =  LumiQueryAPI.deliveredLumiForRange (session, parameters, inputRange)    
        if not options.outputfile:
             LumiQueryAPI.printDeliveredLumi (lumidata, '')
        else:
            lumidata.insert (0, ['run', 'nls', 'delivered', 'beammode'])
            LumiQueryAPI.dumpData (lumidata, options.outputfile)

    # Recorded
    if options.action ==  'recorded':
        hltpath = ''
        if options.hltpath:
            hltpath = options.hltpath
        lumidata =  LumiQueryAPI.recordedLumiForRange (session, parameters, inputRange)
        if not options.outputfile:
             LumiQueryAPI.printRecordedLumi (lumidata, parameters.verbose, hltpath)
        else:
            todump = dumpRecordedLumi (lumidata, hltpath)
            todump.insert (0, ['run', 'hltpath', 'recorded'])
            LumiQueryAPI.dumpData (todump, options.outputfile)

    # Overview
    if options.action ==  'overview':
        hltpath = ''
        if options.hltpath:
            hltpath = options.hltpath
        delivereddata = LumiQueryAPI.deliveredLumiForRange(session, parameters, inputRange)
        recordeddata  = LumiQueryAPI.recordedLumiForRange(session, parameters, inputRange)
        if not options.outputfile:
            LumiQueryAPI.printOverviewData (delivereddata, recordeddata, hltpath)
        else:
            todump =  LumiQueryAPI.dumpOverview (delivereddata, recordeddata, hltpath)
            if not hltpath:
                hltpath = 'all'
            todump.insert (0, ['run', 'delivered', 'recorded', 'hltpath:'+hltpath])
            LumiQueryAPI.dumpData (todump, options.outputfile)

    # Lumi by lumisection
    if options.action ==  'lumibyls' or options.action == 'lumibylsXing':
        recordeddata  = LumiQueryAPI.recordedLumiForRange  (session, parameters, inputRange)
        # we got it, now we got to decide what to do with it
        if not options.outputfile:
            LumiQueryAPI.printPerLSLumi (recordeddata, parameters.verbose)
        else:
            todump =  LumiQueryAPI.dumpPerLSLumi (recordeddata)
            todump.insert (0, ['run', 'ls', 'delivered', 'recorded'])
            LumiQueryAPI.dumpData (todump, options.outputfile)

    
     
