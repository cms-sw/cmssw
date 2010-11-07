#!/usr/bin/env python
# Colin Bernet, August 2009

from optparse import OptionParser
import sys,os, re, pprint
import castortools

parser = OptionParser()
parser.usage = "%prog <dir1> <dir2> <regexp pattern>: copy all files matching regexp in a castor directory.\n\nExample (just try, the -n option negates the command!):\nrfcpMany.py  /castor/cern.ch/user/c/cbern/CMSSW312/SinglePions /tmp '.*\.root' -n "
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)

(options,args) = parser.parse_args()

if len(args)!=3:
    parser.print_help()
    sys.exit(1)

dir1 = args[0]
dir2 = args[1]
regexp = args[2]


files = castortools.matchingFiles( dir1, regexp,
                                   protocol=False,
                                   castor = castortools.isCastorDir(dir1) )

if options.negate:
    print 'NOT copying ',  
    pprint.pprint(files)
else:
    print 'Copying ',  
    pprint.pprint(files)
    castortools.cp( dir2, files )

print 'from:', dir1
print 'to  :', dir2
