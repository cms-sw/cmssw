#!/usr/bin/env python

import os
import subprocess
import argparse
import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib

# setup argument parser 
parser = argparse.ArgumentParser(description='Merge millePedeMonitor-root-files from eos, that are from the same dataset.', 
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
# positional argument: config file
parser.add_argument('eosDir', 
                    action='store', 
                    help = 'path of the eos directory')
# parse argument
args = parser.parse_args()
eosDir = args.eosDir

# read mps.db
lib = mpslib.jobdatabase()
lib.read_db()
for i in xrange(lib.nJobs):
	print lib.JOBSP3[i]

# count how much jobs there are of each dataset
occurences = []
items      = []
for i in xrange(lib.nJobs):
	if lib.JOBSP3[i] not in items:
		items.append(lib.JOBSP3[i])

for i in xrange(len(items)):
	occurences.append(lib.JOBSP3.count(items[i]))

# copy files from eos and combine root-files of each dataset with "hadd"
eos = '/afs/cern.ch/project/eos/installation/cms/bin/eos.select'
counter = 0
for i in xrange(len(items)):
	command  = 'hadd '
	command += 'monitormerge_'+items[i]+'.root '	
	for j in xrange(occurences[i]):
		os.system(eos+' cp /eos/cms'+eosDir+'/millePedeMonitor%03d.root .' % (counter+j+1))
		command += 'millePedeMonitor%03d.root ' % (counter+j+1)	
	os.system(command)
	for j in xrange(occurences[i]):
		os.system('rm millePedeMonitor%03d.root' % (counter+j+1))
	counter += occurences[i]















