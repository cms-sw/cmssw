#!/usr/bin/env python

from optparse import OptionParser
import sys,os, re, pprint
import castortools
    
parser = OptionParser()
parser.usage = "%prog <castor dir> <regexp pattern 1> <regexp pattern 2>\nPuts in sync the root files corresponding to different outputs of the same process. By default, the single files are detected, but are not removed"
parser.add_option("-d", "--remove-dirty", action="store_true",
                  dest="removeDirty",
                  help="Remove dirty files",
                  default=False)
parser.add_option("-s", "--remove-single", action="store_true",
                  dest="removeSingle",
                  help="Remove single files",
                  default=False)
parser.add_option("-t", "--tolerance",  
                  dest="cleanTolerance",
                  help="relative tolerance on the file size for considering the file. For a tolerance of 0.5, files with a size smaller than 50% of the average size of all files won't be considered.",
                  default="0.05")



(options,args) = parser.parse_args()

if len(args)!=3:
    parser.print_help()
    sys.exit(1)

castorDir = args[0]
regexp1 = args[1]
regexp2 = args[2]

(clean1, dirty1) = cleanFiles( castorDir, regexp1, options.cleanTolerance)
(clean2, dirty2) = cleanFiles( castorDir, regexp2, options.cleanTolerance)

print 'dirty files, 1: '
pprint.pprint( dirty1 )

print 'dirty files, 2: '
pprint.pprint( dirty2 )

if options.removeDirty:
    trash = 'Dirty'
    absTrash = castortools.createSubDir( trash )
    castortools.remove( absTrash, dirty1 )
    castortools.remove( absTrash, dirty2 )
elif len(dirty1) or len(dirty2):
    print 'to remove dirty files in both collections, run again with option -d'

single = castortools.sync( regexp1, clean1, regexp2, clean2 )

if options.removeSingle:
    trash = 'Single'
    absTrash = castortools.createSubDir( trash )
    castortools.remove( absTrash, single )
elif len(single):
    print 'to remove single files in both collections, run again with option -s'

