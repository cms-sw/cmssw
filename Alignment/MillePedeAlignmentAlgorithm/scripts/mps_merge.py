#!/usr/bin/env python
#
#  produce cfg file for merging run
#
#  Usage:
#
#  mps_merge.pl [-c] inCfg mergeCfg mergeDir njobs

import Alignment.MillePedeAlignmentAlgorithm.mpslib.Mpslibclass as mpslib
import re
import os
import argparse

lib = mpslib.jobdatabase()

## Parse arguments
## -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Produce Config for Pede-job from Mille-config.')

# optional arguments
parser.add_argument('-c', '--checkok', action='store_true',
                    help='check which jobs are OK and write just them to the config')
parser.add_argument('-w', '--checkweight', action='store_true',
                    help='check for weight assignments in mps.db and add them to binarylist')
parser.add_argument("-a", "--append", dest="append", metavar="SNIPPET",
                    help="config snippet to be appended to the output config")

# positional arguments
parser.add_argument('inCfg', action='store',
                    help='path to cfg-file, that is used as base')
parser.add_argument('mergeCfg', action='store',
                    help='path and name of the config that is produced')
parser.add_argument('mergeDir', action='store',
                    help='path to the merge directory')
parser.add_argument('nJobs', action='store', type=int,
                    help='number of jobs')

# parse arguments
args = parser.parse_args()
inCfg       = args.inCfg
mergeCfg    = args.mergeCfg
mergeDir    = args.mergeDir
nJobs       = args.nJobs
checkok     = args.checkok
checkweight = args.checkweight

if checkok or checkweight:
    lib.read_db()

## create merge config
## -----------------------------------------------------------------------------

# create pede dir
if not os.path.isdir(mergeDir):
    os.system('mkdir '+mergeDir)

# open base config
with open(inCfg, 'r') as INFILE:
    body = INFILE.read()


# set mode to "pede"
match = re.search('setupAlgoMode\s*?\=\s*?[\"\'].*?[\"\']', body)
if match:
    body = re.sub('setupAlgoMode\s*?\=\s*?[\"\'].*?[\"\']',
                  'setupAlgoMode = \"pede\"',
                  body)
else:
    print 'Error in mps_merge: No setupAlgoMode found in baseconfig.'


# build list of binary files
binaryList = ''
firstentry = True
for i in xrange(nJobs):
    separator = ',\n                '
    if firstentry:
        separator = '\n                '
    if checkok and lib.JOBSTATUS[i]!='OK':
        continue
    firstentry = False

    newName = 'milleBinary%03d.dat' % (i+1)
    if checkweight and (lib.JOBSP2[i]!='' and lib.JOBSP2[i]!='1.0'):
        weight = lib.JOBSP2[i]
        print 'Adding %s to list of binary files using weight %s' % (newName,weight)
        binaryList = '%s%s\'%s -- %s\'' % (binaryList, separator, newName, weight)
    else:
        print 'Adding %s to list of binary files' % newName
        binaryList = '%s%s\'%s\'' % (binaryList, separator, newName)


# replace 'placeholder_binaryList' with binaryList
match = re.search('[\"\']placeholder_binaryList[\"\']', body)
if match:
    body = re.sub('[\"\']placeholder_binaryList[\"\']',
                  binaryList,
                  body)
else:
    print 'Error in mps_merge: No \'placeholder_binaryList\' found in baseconfig.'


# build list of tree files
treeList =''
firstentry = True
for i in xrange(nJobs):
    separator = ',\n                '
    if firstentry:
        separator = '\n                '
    if checkok and lib.JOBSTATUS[i]!='OK':
        continue
    firstentry = False

    newName = 'treeFile%03d.root' % (i+1)
    print 'Adding %s to list of tree files.' % newName
    treeList = '%s%s\'%s\'' % (treeList, separator, newName)


# replace 'placeholder_treeList' with binaryList
match = re.search('[\"\']placeholder_treeList[\"\']', body)
if match:
    body = re.sub('[\"\']placeholder_treeList[\"\']',
                  treeList,
                  body)
else:
    print 'Error in mps_merge: No \'placeholder_treeList\' found in baseconfig.'

if args.append is not None:
    with open(args.append) as snippet:
        body += snippet.read()

with open(mergeCfg, 'w') as OUTFILE:
    OUTFILE.write(body)
