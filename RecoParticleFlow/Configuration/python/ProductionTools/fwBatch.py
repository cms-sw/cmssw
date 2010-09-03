#!/usr/bin/env python
# Colin
# additional layer, on top of cmsBatch.py

import os, sys,  imp, re, pprint, string
from optparse import OptionParser

import castortools

parser = OptionParser()
parser.usage = "fwBatch.py <sampleName> <extension>"
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

if len(args)!=2:
    parser.print_help()
    sys.exit(1)

sampleName = args[0]
ext = args[1]


print 'starting prod for sample:', sampleName

# preparing castor dir -----------------

cdir = options.castorBaseDir 
cdir += sampleName

outFile = cdir
outFile += '/PFAnalysis_%s.root' % ext

cmsBatch = 'cmsBatch.py 4 jetMET_cfg.py -p jetMETAnalysis -r %s -b "bsub -q cmst3 <  batchScript.sh"' % outFile

print cmsBatch
