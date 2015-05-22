#!/usr/bin/env python
# Colin
# creates new source file for a dataset on CAF
# compiles the python module
# prints the line to be added to the cfg. 

import os, sys,  imp, re, pprint, string, subprocess
from optparse import OptionParser

import CMGTools.Production.castortools as castortools

parser = OptionParser()
parser.usage = """
importNewSource.py <sampleName>
Create the source file corresponding to a given sample on CAF. Run it from the package where you want to put the source cff, for example CMGTools/Common.

For example, the source file for /HT/Run2011A-May10ReReco-v1/AOD
would be placed in
python/sources/HT/Run2011A_May10ReReco_v1/AOD/source_cff.py
and can easily be loaded in any cfg.

Note that the script makes sure to change all '-' into '_' when creating the destanation directory from the dataset name, so that the source module can then be loaded in a python cfg. 
"""

parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)
parser.add_option("-w", "--wildcard", 
                  dest="wildcard",
                  help="UNIX style wildcard for root files in castor dir.",
                  default="*root")
parser.add_option("-o", "--output", 
                  dest="output",
                  help="Output file name.",
                  default="source_cff.py")

(options,args) = parser.parse_args()

if len(args)!=1:
    parser.print_help()
    sys.exit(1)

sampleName = args[0].rstrip('/')

# getting all files from this sample -----------------

dbs = 'dbs search --query="find file where dataset like %s"' % sampleName

dbsOut = os.popen(dbs)

# allFiles = []
for line in dbsOut:
    if line.find('/store')==-1:
        continue
    line = line.rstrip()
    # print 'line',line
    # allFiles.append(line)


from CMGTools.Production.doImportNewSource import doImportNewSource
doImportNewSource( sampleName, 'sourceFileListCAF.py ' + sampleName, options.output)
