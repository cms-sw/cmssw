#!/usr/bin/env python
# Colin
# creates new source file for a dataset on castor
# compiles the python module
# prints the line to be added to the cfg. 

import os, sys,  imp, re, pprint, string
from optparse import OptionParser

import castortools

parser = OptionParser()
parser.usage = "importNewSource.py <sampleName>\nCreate the source file corresponding to a given sample on castor. Run it from the package where you want to put the source cff."
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




print 'starting prod for sample:', sampleName

# preparing castor dir -----------------

cdir = options.castorBaseDir 
cdir += sampleName

# making local source directory ---------

tmp = './python/Sources' + sampleName
ldir = re.sub( '-', '_', tmp)
mkdir = 'mkdir -p ' + ldir
print mkdir
os.system( mkdir )

# creating source file ------------------

sourceFile = ldir + '/source_cff.py'
sourceFileList = 'sourceFileList.py %s "%s" > %s' % (cdir, options.pattern, sourceFile)
print sourceFileList
os.system(sourceFileList)
# compile new source file
os.system( 'scram b python')

# printing module load command ----------
base = os.environ['CMSSW_BASE']
cwd = os.getcwd()
cwd = re.sub('%s/src/' % base, '', cwd)
# cwd now equal to package name

# replace ./python by package name 
module = re.sub( './python', cwd, sourceFile)
# replace / by .
module = re.sub( '/', '.', module)
# remove trailing .py
module = re.sub( '\.py$', '', module)

os.system( 'cat %s' % sourceFile )

print 'new source file ready to be used:'
print sourceFile
print 'process.load("%s")' % module
