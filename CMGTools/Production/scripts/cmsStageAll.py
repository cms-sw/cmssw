#!/usr/bin/env python
# Colin Bernet, August 2009

from optparse import OptionParser
import sys,os, re, pprint
import CMGTools.Production.eostools as castortools

parser = OptionParser()
parser.usage = """
%prog <dir1> <dir2> <regexp pattern>: copy all files matching regexp in a castor directory.

Example (just try, the -n option negates the command!):\ncmsStageAll.py /store/cmst3/user/cbern/CMG/HT/Run2011A-PromptReco-v1/AOD/PAT_CMG /tmp '.*\.root' -n\n\nIMPORTANT NOTE: castor directories must be provided as logical file names (LFN), starting by /store."""

parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)
parser.add_option("-f", "--force", action="store_true",
                  dest="force",
                  help="force overwrite",
                  default=False)


(options,args) = parser.parse_args()

if len(args)!=3:
    parser.print_help()
    sys.exit(1)

dir1 = args[0]
dir2 = args[1]
regexp = args[2]


files = castortools.matchingFiles( dir1, regexp )

if options.negate:
    print 'NOT copying ',  
    pprint.pprint(files)
else:
    print 'Copying ',  
    pprint.pprint(files)

    castortools.cmsStage( dir2, files, options.force) 
    
print 'from:', dir1
print 'to  :', dir2
