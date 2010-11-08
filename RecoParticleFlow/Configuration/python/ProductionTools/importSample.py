#!/usr/bin/env python
# Colin
# creates new source file for a dataset on castor
# compiles the python module
# prints the line to be added to the cfg. 

import os, sys,  imp, re, pprint, string
from optparse import OptionParser

import castortools

parser = OptionParser()
parser.usage = "%prog <sampleName>\nImport a sample locally."
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)
parser.add_option("-c", "--castorBaseDir", 
                  dest="castorBaseDir",
                  help="Base castor directory. Subdirectories will be created automatically for each prod",
                  default="/castor/cern.ch/user/c/cbern/cmst3/SusyJetMET")
parser.add_option("-p", "--pattern", 
                  dest="pattern",
                  help="pattern for root files in castor dir",
                  default=".*root")

(options,args) = parser.parse_args()

if len(args)!=1:
    parser.print_help()
    sys.exit(1)

sampleName = args[0]

pattern = options.pattern



print 'starting prod for sample:', sampleName

# preparing castor dir -----------------

cdir = options.castorBaseDir
cdir += sampleName

# making local source directory ---------

ldir = "./"+sampleName

mkdir = 'mkdir -p ' + ldir
print mkdir
os.system( mkdir )

# copy

rfcp = 'rfcpMany.py %s %s "%s"' % ( cdir, ldir, pattern ) 
print rfcp

