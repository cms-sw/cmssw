#!/usr/bin/env python
# colin
# additional layer, on top of cmsBatch.py

import os, sys,  imp, re, pprint, string
from optparse import OptionParser

import CMGTools.Production.castortools as castortools
from CMGTools.Production.addToDatasets import *

parser = OptionParser()
parser.usage = """
fwBatch.py <cfg> <sampleName>:
Additional layer on top of cmsBatch.py (see the help of this script for more information). This script simply prepares a cmsBatch.py command for you to run. 

Example:

First do:
cd $CMSSW_BASE/src/CMGTools/Common/test

fwBatch.py -N 1 testCMGTools_cfg.py /HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG_MAX -b 'nohup ./batchScript.sh &' -t Test

output:
starting prod for sample: /HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG_MAX
sampleName  /HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG_MAX/Test
mkdir -p .//HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG_MAX/Test
/HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG_MAX/Test
cmsBatch.py 1 testCMGTools_cfg.py -r /store/cmst3/user/cbern/CMG/HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG_MAX/Test -b 'nohup ./batchScript.sh &' -o .//HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG_MAX/Test

Then, just run this cmsBatch.py command.

IMPORTANT: make sure that the source you are reading in testCMGTools_cfg.py corresponds to the sample you specify!!!
"""



parser.add_option("-t", "--tier", 
                  dest="tier",
                  help="Tier: extension you can give to specify you are doing a new production. If you give a Tier, your new files will appear in sampleName/tierName, which will constitute a new dataset.",
                  default="")
parser.add_option("-N", "--numberOfInputFiles", 
                  dest="nInput",
                  help="Number of input files per job",
                  default="5")
parser.add_option("-b", "--batch", 
                  dest="batch",
                  help="Batch command. Same as in cmsBatch.py",
                  default="bsub -q 8nh < batchScript.sh")


(options,args) = parser.parse_args()

if len(args)!=2:
    parser.print_help()
    sys.exit(1)

cfg = args[0]
sampleName = args[1]

import castorBaseDir
destBaseDir = castorBaseDir.myCastorBaseDir()

#if options.castorBaseDir.find('/castor/cern.ch/user/c/cbern') == -1:
#    destBaseDir = castorBaseDir.defaultCastorBaseDir


print 'starting prod for sample:', sampleName

# preparing castor dir -----------------

# cdir = options.castorBaseDir 

if options.tier != "":
    sampleName += "/" + options.tier

print "sampleName ",sampleName
outFile = destBaseDir
outFile += sampleName

# prepare local output dir:
localOutputDir = './' + sampleName 
mkdir = 'mkdir -p ' + localOutputDir
print mkdir
os.system(mkdir)

# the output castor directory will be prepared by cmsBatch

cmsBatch = 'cmsBatch.py %s %s -r %s -b "%s" -o %s' % (options.nInput, cfg, outFile, options.batch, localOutputDir)

addToDatasets( sampleName )


print cmsBatch
