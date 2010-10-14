#!/usr/bin/env python

from optparse import OptionParser
import sys,os, re, pprint
import castortools

parser = OptionParser()
parser.usage = "%prog <castor dir> <regexp pattern>: place all empty files in a trash. This script is based on edmFileUtil, so it's pretty slow. The files will be moved to the trash only at the end of the processing. Does anybody know of a fast way to get the number of events in an EDM file? If yes contact Colin.\n\nExample (just try, the -n option negates the command!):\nremoveEmptyFiles.py  /castor/cern.ch/user/c/cbern/CMSSW312/SinglePions '.*\.root' -n"
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)

(options,args) = parser.parse_args()

if len(args)!=2:
    parser.print_help()
    sys.exit(1)

castorDir = args[0]
regexp = args[1]

if options.negate:
    print 'files will NOT be removed'

files = castortools.emptyFiles( castorDir, regexp,
                                castortools.isCastorDir(castorDir) )

if options.negate:
    print 'NOT removing ',  
    pprint.pprint(files)
else:
    print 'Removing ',  
    pprint.pprint(files)
    trash = castortools.createSubDir( castorDir, 'Trash')
    print trash
    castortools.move( trash, files )
