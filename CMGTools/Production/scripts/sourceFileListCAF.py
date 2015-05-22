#!/usr/bin/env python

from optparse import OptionParser
import sys,os

parser = OptionParser()
parser.usage = "%prog <dataset_on_CAF> : format the ROOT files in a given dataset on CAF as a source module for cmsRun. \n\nExample (just try!):\nsourceFileListCAF.py /DoubleMu/Run2011A-ZMu-PromptSkim-v4/RAW-RECO"


(options,args) = parser.parse_args()


if len(args) != 1:
    parser.print_help()
    sys.exit(1)

sampleName = args[0].rstrip('/')

dbs = 'dbs search --query="find file where dataset like %s"' % sampleName

dbsOut = os.popen(dbs)

allFiles = []
for line in dbsOut:
    if line.find('/store')==-1:
        continue
    line = line.rstrip()
    # print 'line',line
    allFiles.append(line)


from sourceFileListCff import sourceFileListCff
print sourceFileListCff( allFiles )

