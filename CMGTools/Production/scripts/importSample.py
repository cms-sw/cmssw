#!/usr/bin/env python
# Colin
# creates new source file for a dataset on castor
# compiles the python module
# prints the line to be added to the cfg. 

import os, sys,  imp, re, pprint, string, fnmatch
from optparse import OptionParser

import CMGTools.Production.eostools as castortools

parser = OptionParser()
parser.usage = "%prog <sampleName>\nImport a sample locally."
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)


#parser.add_option("-c", "--castorBaseDir", 
#                  dest="castorBaseDir",
#                  help="Base castor directory. Subdirectories will be created automatically for each prod",
#                  default=castorBaseDir.defaultCastorBaseDir)
parser.add_option("-u", "--user", 
                  dest="user",
                  help="user who is the owner of the castor base directory",
                  default=os.environ['USER'] )
parser.add_option("-w", "--wildcard", 
                  dest="wildcard",
                  help="UNIX style wildcard for root files in castor dir",
                  default=".*root")

(options,args) = parser.parse_args()

if len(args)!=1:
    parser.print_help()
    sys.exit(1)

sampleName = args[0]

pattern = fnmatch.translate( options.wildcard )


# preparing castor dir -----------------

import CMGTools.Production.castorBaseDir as castorBaseDir

try:
    cdir = castorBaseDir.castorBaseDir( options.user ) 
except:
    print 'user does not have a castor base dir'
    sys.exit(1)
    
cdir += sampleName

if not castortools.fileExists( cdir ):
    print 'Directory ', cdir, 'does not exist'
    print 'Please check the sample name, and the user. You can specify a different user using the -u option'
    sys.exit(2)

# making local source directory ---------

ldir = "./"+sampleName

mkdir = 'mkdir -p ' + ldir
print mkdir

if not options.negate:
    os.system( mkdir )

# copy

cmsStage = 'cmsStageAll.py -f %s %s "%s" ' % ( cdir, ldir, pattern ) 
if options.negate:
    cmsStage += '-n'

print cmsStage


os.system( cmsStage )
