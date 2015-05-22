#!/usr/bin/env python

from optparse import OptionParser
import sys,os, re, pprint
import CMGTools.Production.castortools as castortools

parser = OptionParser()
parser.usage = """
%prog <castor dir> <regexp pattern>: place all files matching regexp in a castor directory in a Trash.

Example (just try, the -n option negates the command!):
rfrmMany.py /store/cmst3/user/cbern/CMG/HT/Run2011A-PromptReco-v1/AOD/PAT_CMG '.*\.root' -n
IMPORTANT NOTE: castor directories must be provided as logical file names (LFN), starting by /store.
"""

parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)
parser.add_option("-k", "--kill", action="store_true",
                  dest="kill",
                  help="really remove the files",
                  default=False)

(options,args) = parser.parse_args()

if len(args)!=2:
    parser.print_help()
    sys.exit(1)

castorDir = args[0]
regexp = args[1]

files = castortools.matchingFiles( castorDir, regexp )

if options.negate:
    print 'NOT removing ',  
    pprint.pprint(files)
else:
    if options.kill == False:
        pprint.pprint(files)
        trash = castortools.createSubDir( castorDir, 'Trash')
        castortools.move( trash, files )
    else:
        castortools.remove( files )
