#! /usr/bin/env python
"""
runPixelPopConCalib.py

Python script to run PixelPopConCalibAnalyzer application

Initial version: M. Eads, Sep 2008
"""

import os, sys, getopt, shutil
from socket import getfqdn

def usage():
    print """
runPixelPopConCalib.py usage:
    
This script runs the pixel popcon application for the calibration
configuration object. It accepts the following arguments

-h (or --help)  prints this usage message
-d (or --debug) sets to debug mode, printing extra information
-f XXX (or --filename=XXX) filename (and path to) calib.dat file (REQUIRED)
-r XXX (or --runnumber=XXX) run number, used to set IOV (REQUIRED)
-t XXX (or --tagname=XXX) tag name to use (by default, will use default based on calibration type)
-c XXX (or --cfgtemplate=XXX) template cfg.py file to use (defaults to CondTools/SiPixel/test/test_PixelPopConCalibAnalyzer_cfg.py)
-D XXX (or --database=XXX) connect string for database. Defaults to ORCON if at P5, sqlite file otherwise
-l XXX (or --logdatabase=XXX) connect string for logging database. Defaults to official logging db if at P5, sqlite file otherwise
-a XXX (or --authPath=XXX) path to database authentication files
-p (or --point5) Force point5 behavior (this gets set by default if running from a machine in the .cms network)
-o XXX (or --outputFilename=XXX) Filename for output cfg.py file
-w (or --writeOnly) Only write the cfg.py file, don't run it
-q (or --writeChecker) Also write a PixelPopConCalibChecker cfg.py file
-Q XXX (or --writeCheckerTemplate=XXX) Template cfg.py for PixelPopConCalibChecker 
"""

def main(argv):
    # check that CMSSW environment has been set up
    if 'CMSSW_BASE' not in os.environ:
        print 'CMMSW is not set up! Please run "cmsenv"'
        sys.exit(2)
    
    # get the options from the command line
    try:
        opts, args = getopt.getopt(argv, 'hdf:t:r:c:D:l:a:po:wqQ:', 
                                   ['help', 'debug', 'filename=', 'tagname=', 'runnumber=', 
                                    'cfgtemplate=', 'database=', 'logdatabase=', 'authPath=', 
                                    'point5', 'outputFilename=', 'writeOnly', 'writeChecker',
                                    'writeCheckerTemplate='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
        
    # if no options given, print usage and exit
    if not opts:
        print 'runPixelPopConCalib.py: No options given'
        usage()
        sys.exit(2)
    
    #print 'opts:', opts
    
    # figure out if we are at point 5
    atPoint5 = False
    hostname = getfqdn()
    if hostname.split('.')[-1] == 'cms':
        atPoint5 = True
        
    debugMode = False
    calibFilename = False
    tagName = False
    runNumber = False
    cfgTemplate = False
    databaseConnect = False
    logdbConnect = False
    authenticationPath = False
    writeOnly = False
    writeFilename = False
    writeChecker = False
    writeCheckerTemplate = False
    writeCheckerFilename = False
    # loop over command-line arguments
    for opt, value in opts:
        if opt in ('-h', '--help'):
            usage()
            sys.exit()
        elif opt in ('-d', '--debug'):
            debugMode = True
        elif opt in ('-f', '--filename'):
            calibFilename = value
        elif opt in ('-t', '--tagname'):
            tagName = value
        elif opt in ['-r', '--runnumber']:
            # check that it's an integer
            if not value.isdigit():
                print 'Run number given was', value, ', which is not an integer'
                sys.exit(2)
            runNumber = value
        elif opt in ['-c', '--cfgtemplate']:
            cfgTemplate = value
        elif opt in ['-D', '--database']:
            databaseConnect = value
        elif opt in ['-l', '--logdatabase']:
            logdbConnect = value 
        elif opt in ['-a', '--authPath']:
            authenticationPath = value
        elif opt in ['-p', '--point5']:
            atPoint5 = True
            if debugMode:
                print '** forcing point5 mode'
        elif opt in ['-o', '--outputFilename']:
            writeFilename = value
        elif opt in ['-w', '--writeOnly']:
            writeOnly = True
        elif opt in ['-q', '--writeChecker']:
            writeChecker = True
        elif opt in ['-Q', '--writeCheckerTemplate']:
            writeCheckerTemplate = value
            
    
    if debugMode:
        print '** debugMode activated'
        
    # check that calib filename was provided
    if not calibFilename:
        print 'You must provide a path to the calib.dat file with the -f (or ---filename) option'
        sys.exit(2)
    if debugMode:
        print '** calib.dat filename set to', calibFilename
    
    # set the tagname if not provided
    if not tagName:
    	tagName = getTagNameFromFile(calibFilename, debugMode)
    	if not tagName:
        	print 'Unknown calibration type from calib.dat file!'
        	sys.exit(2)
    if debugMode:
        print '** tag name set to', tagName
        
    # check that the run number was provided
    if not runNumber:
        print 'You must provide a run number to set the IOV'
        sys.exit(2)
    if debugMode:
        print '** run number for IOV set to', runNumber
        
    # set cfg template to default if not given
    if not cfgTemplate:
        cfgTemplate = os.environ['CMSSW_BASE'] + '/src/CondTools/SiPixel/test/testPixelPopConCalibAnalyzer_cfg.py'
    if debugMode:
        print '** Using cfg file template:', cfgTemplate
        
    if atPoint5:
        print '** point 5 mode is set'
        
    # set database connect string if not given
    if not databaseConnect:
        if atPoint5:
            databaseConnect = 'oracle://cms_orcon_prod/CMS_COND_31X_PIXEL'
        else:
            databaseConnect = 'sqlite_file:testExample.db'
    if debugMode:
        print '** database connect string:', databaseConnect
        
    # set the logging database connect string if not given
    if not logdbConnect:
        if atPoint5:
            logdbConnect = 'oracle://cms_orcon_prod/CMS_COND_31X_POPCONLOG'
        else:
            logdbConnect = 'sqlite_file:logtestExample.db'
    if debugMode:
        print '** logging db connect string:', logdbConnect
        
    if not authenticationPath:
        if atPoint5:
            authenticationPath = '/nfshome0/xiezhen/conddb'
        else:
            authenticationPath = '/afs/cern.ch/cms/DB/conddb'
        
    if writeOnly and debugMode:
        print '** PixelPopConCalib cfg file will only be written, not run'
        
    if not writeFilename:
        writeFilename = 'PixelPopConCalib_' + tagName + '_' + runNumber + '_cfg.py'
    if debugMode:
        print '** PixelPopConCalib cfg file will be named ', writeFilename
        
    if writeChecker:
        if not writeCheckerFilename:
            writeCheckerFilename = 'PixelPopConCalibChecker_' + tagName + '_' + runNumber + '_cfg.py'
        if not writeCheckerTemplate:
            writeCheckerTemplate = os.environ['CMSSW_BASE'] + '/src/CondTools/SiPixel/test/PixelPopConCalibChecker_cfg.py'
        if debugMode:
            print '** PixelPopConCalibChecker cfg file will be written from template', writeCheckerTemplate
            print '   with filename', writeCheckerFilename
    
            
    # write the cfg.py file
    writePixelPopConCalibCfg(filename = writeFilename,
                             calibFilename = calibFilename,
                             cfgTemplate = cfgTemplate, 
                             runNumber = runNumber,
                             tagName = tagName,
                             databaseConnect = databaseConnect,
                             logdbConnect = logdbConnect,
                             authenticationPath = authenticationPath,
                             debugMode = debugMode)
    
    # write the checker
    if writeChecker: 
        writePixelPopConCalibCheckerCfg(filename = writeCheckerFilename, 
                                        cfgTemplate = writeCheckerTemplate,
                                        calibFilename = calibFilename,
                                        runNumber = runNumber,
                                        tagName = tagName,
                                        databaseConnect = databaseConnect,
                                        authenticationPath = authenticationPath,
                                        debugMode = debugMode)
    
    # run the popcon calib job
    if not writeOnly:
        if debugMode:
            print '** running the cfg file ', writeFilename
        os.system('cmsRun ' + writeFilename)
    else:
        print 'PixelPopConCalib cfg.py written as', writeFilename
        
    if writeChecker:
        print 'PixelPopConCalibChecker cfg written as', writeCheckerFilename
        print 'To check if the popcon transfer was successful, run "cmsRun "' + writeCheckerFilename
            
def getTagNameFromFile(filename, debugMode = False):
    """
    getTagNameFromFile() reads a calib.dat text file and sets the database calib.dat tag based on the "Mode:" setting
    """
    # open the calib.dat file and find the Mode: line
    if debugMode:
        print '** getting tag name from calib.dat file'
        print '   calib.dat filename:', filename 
    f = open(filename)
    for line in f:
        if line.find('Mode:') == 0:
            if debugMode:
                print '   using line:', line
            if line.find('GainCalibration') != -1:
                return 'GainCalibration_default'
            elif line.find('SCurve') != -1:
                return 'SCurve_default'
            elif line.find('PixelAlive') != -1:
                return 'PixelAlive_default'
            # otherwise, it is an unknown calibration type, return False
            return False
        
def writePixelPopConCalibCfg(filename, cfgTemplate, calibFilename = '',
                             runNumber = '', tagName = '', 
                             databaseConnect = '', logdbConnect = '', 
                             authenticationPath = '', debugMode = False):
    """
    writePixelPopConCalibCfg() writes a cfg.py file to run the PixelPopConCalibAnalyzer job
    """
    # copy the template file to the new cfg.py file
    shutil.copyfile(cfgTemplate, filename)
    
    # open the new cfg file and add the necessary lines
    f = open(filename, 'a')
    f.write('\n')

    if calibFilename:
        f.write('process.PixelPopConCalibAnalyzer.Source.connectString = "file://' + calibFilename + '"\n')                
    if runNumber:
        f.write('process.PixelPopConCalibAnalyzer.Source.sinceIOV = ' + runNumber + '\n')
    if logdbConnect:
        f.write('process.PoolDBOutputService.logconnect = "' + logdbConnect + '"\n')
    if tagName:
        f.write('process.PoolDBOutputService.toPut[0].tag = "' + tagName + '"\n')
    if databaseConnect:
        f.write('process.PoolDBOutputService.connect = "' + databaseConnect + '"\n')
    if authenticationPath:
        f.write('process.PoolDBOutputService.DBParameters.authenticationPath = "' + authenticationPath + '"\n')
        
    

def writePixelPopConCalibCheckerCfg(filename, cfgTemplate, calibFilename = '', runNumber = '', 
                                    tagName = '', databaseConnect = '', 
                                    authenticationPath = '', debugMode = False):
    """
    writePixelPopConCalibCheckerCfg() writes a cfg.py file to run a PixelPopConCalibChecker job
    """
    # copy the template file to the new cfg.py file
    shutil.copyfile(cfgTemplate, filename)
    
    # open the new cfg file and add the necessary lines
    f = open(filename, 'a')
    f.write('\n')
    
    if calibFilename:
        f.write('process.demo.filename = "' + calibFilename + '"\n')
    if runNumber:
        f.write('process.source.firstValue = ' + runNumber + '\n')
        f.write('process.source.lastValue = ' + runNumber + '\n')
    if tagName:
        f.write('process.sipixelcalib_essource.toGet[0].tag = "' + tagName + '"\n')
    if databaseConnect:
        f.write('process.sipixelcalib_essource.connect = "' + databaseConnect + '"\n')
    if authenticationPath:
        f.write('process.sipixelcalib_essource.DBParameters.authenticationPath = "' + authenticationPath + '"\n')
        

if __name__ == '__main__':
    main(sys.argv[1:])
