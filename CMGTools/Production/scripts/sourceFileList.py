#!/usr/bin/env python

from optparse import OptionParser
import sys,os, pprint, re
import CMGTools.Production.eostools as castortools

    
parser = OptionParser()
parser.usage = "%prog <dir> <regexp> : format a set of root files matching a regexp in a directory, as an input to the PoolSource. \n\nExample (just try!):\nsourceFileList.py /castor/cern.ch/user/c/cbern/CMSSW312/SinglePions '.*\.root'"
parser.add_option("-c", "--check", dest="check", default=False, action='store_true',help='Check filemask if available')

(options,args) = parser.parse_args()


if len(args) != 2:
    parser.print_help()
    sys.exit(1)

dir = args[0]
regexp = args[1]


exists = castortools.fileExists( dir )
if not exists:
    print 'sourceFileList: directory does not exist. Exiting'
    sys.exit(1)


files = castortools.matchingFiles( dir, regexp)

mask = "IntegrityCheck"
file_mask = []  
if options.check:
    file_mask = castortools.matchingFiles(dir, '^%s_.*\.txt$' % mask)
bad_files = {}    
if options.check and file_mask:
    from CMGTools.Production.edmIntegrityCheck import PublishToFileSystem
    p = PublishToFileSystem(mask)
    report = p.get(dir)
    if report is not None and report:
        dup = report.get('ValidDuplicates',{})
        for name, status in report['Files'].iteritems():
            if not status[0]:
                bad_files[name] = 'MarkedBad'
            elif dup.has_key(name):
                bad_files[name] = 'ValidDup'


from CMGTools.Production.sourceFileListCff import sourceFileListCff
print sourceFileListCff( files, bad_files)

