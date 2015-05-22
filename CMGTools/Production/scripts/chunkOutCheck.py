#!/bin/env python

import sys
import re
import os
import pprint
import glob

from optparse import OptionParser

parser = OptionParser(usage='%prog <target_directories> [options]',
                      description='Check one or more chunck folders. Wildcard (*) can be used to specify multiple directories')

parser.add_option("-b","--batch", dest="batch",
                  default=None,
                  help="batch command for resubmission"
                  )
parser.add_option("-c","--copy-integrity", dest="checkCopy",
                  default=False, action="store_true",
                  help="check copy integrity from output of batch job. File to check given by -l argument"
                  )
parser.add_option("-l","--log-file", dest="logfile",
                  default="",
                  help="if option -c check copy integrity in given log-file (default: check *.log *.txt *.out STD_OUTPUT)"
                  )

(options,args) = parser.parse_args()

if len(args)==0:
    print 'provide at least one directory in argument. Use -h to display help'

dirs = sys.argv[1:]

badDirs = []

for d in dirs:
    if not os.path.isdir(d):
        continue
    if d.find('_Chunk') == -1:
        continue
    logName  = '/'.join([d, 'log.txt'])
    if not os.path.isfile( logName ):
        print d, ': log.txt does not exist'
        badDirs.append(d)
        continue
    logFile = open(logName)
    nEvents = -1
    for line in logFile:
        try:
            nEvents = line.split('processed:')[1]
        except:
            pass
    if nEvents == -1:
        print d, 'cannot find number of processed events'
    elif nEvents == 0:
        print d, '0 events'
    else:
        #everything ok so far
        if options.checkCopy:
            match = ["*.txt", "*.log","*.out", "STD_OUTPUT"] if options.logfile == "" else [options.logfile]
            logNames = []
            for m in match:
                logNames += glob.glob(d+"/"+m)
            succeeded = False
            for logName in logNames:
                if not os.path.isfile( logName ):
                    print logName, 'does not exist'
                else:
                    logFile = open(logName)
                    isRemote = False
                    for line in logFile:
                        if "gfal-copy" in line:
                            isRemote = True
                        if "copy succeeded" in line and "echo" not in line:
                            if not isRemote or "remote" in line:
                                succeeded = True
                                break
                            else:
                                print logName, ': remote copy failed. Copied locally'
                if succeeded:
                    break
            if succeeded:
                continue # all good
            if logNames == []:
                print d, ": no log files found matchig", match
            else:
                print logNames, ': copy failed or not sent to the expected location'
        else:
            continue # all good
    badDirs.append(d)

print 'list of bad directories:'
pprint.pprint(badDirs)

if options.batch is not None:
    for d in badDirs:
        oldPwd = os.getcwd()
        os.chdir( d )
        cmd =  [options.batch, '-J', d, ' < batchScript.sh' ]
        print 'resubmitting in', os.getcwd()
        cmds = ' '.join( cmd )
        print cmds
        os.system( cmds )
        os.chdir( oldPwd )
        
