#!/usr/bin/env python
# Colin
# creates new source file for a dataset on castor
# compiles the python module
# prints the line to be added to the cfg. 

import os, sys, re
from optparse import OptionParser

import CMGTools.Production.eostools as castortools


def allSampleInfo( sampleName, listLevel ):

    if listLevel == 3:
        contents = castortools.ls(castorDir)
        for c in contents:
            print c
        #os.system("rfdir %s | awk '{print \"%s/\"$9}'" % (castorDir,castorDir) )
        return

    print sampleName

    if listLevel>0:
        print '------------------------------------------------------------------------------------------------------------'
        print 'PFN:'
        print castorDir
        print 'LFN:'
        print castortools.castorToLFN(castorDir)
    if listLevel>1:
        contents = castortools.ls(castorDir)
        for c in contents:
            print c
    if listLevel>0 and localDir!=None:
        print 'local:'
        print localDir
        if os.path.isdir( localDir ):
            if listLevel>1:
                os.system('ls -l ' + localDir )
                # print localDir + '*.root'
        else:
            if listLevel>0:
                print 'TO BE IMPORTED'
    if listLevel>0:
        print
        print


parser = OptionParser()
parser.usage = """
%prog <sampleName>
List datasets.

It is advisable to import some of your datasets locally.
In this case, choose a local base directory where you will import your datasets, somewhere where you have space. You can import your datasets locally using importSample.py from this directory.
Set the following environment variable so that listSamples.py knows where to find your local samples:

export CMGLOCALBASEDIR=<your local base dir>

Examples:
listSamples.py /HT/Run2011A-May10ReReco-v1/AOD/BADPF -u cbern
listSamples.py /HT/Run2011A-May10ReReco-v1/AOD/BADPF -u cbern -l 2
"""

import CMGTools.Production.castorBaseDir as cBaseDir

parser.add_option("-u", "--user", 
                  dest="user",
                  help="User who is the owner of the castor base directory. Note that this user must have his/her ~/public/DataSets.txt up to date",
                  default=os.environ['USER'] )
#parser.add_option("-d", "--localBaseDir", 
#                  dest="localBaseDir",
#                  help="Local base directory. In case you have a local base directory where you import your samples, you can",
#                  default="/afs/cern.ch/user/c/cbern/localscratch/Data/Analysis/SusyJetMET")
parser.add_option("-l", "--listLevel", 
                  dest="listLevel", 
                  help="list level",
                  default=False)

(options,args) = parser.parse_args()

if len(args)!=1:
    parser.print_help()
    sys.exit(1)


castorDir = ""
localDir = ""

# opions.user could be of the form user_area
user,area = cBaseDir.getUserAndArea(options.user)

dataSets = '/afs/cern.ch/user/{first}/{user}/public/DataSets.txt'.format(
    first = user[0], # first letter of the username
    user = user
    )

ifile=open(dataSets,'r')

pattern = re.compile( args[0] )

castorBaseDir = castortools.lfnToCastor(cBaseDir.castorBaseDir( options.user ))

for line in ifile.readlines():
    line = line.rstrip()
    if len(line)==0 or line[0]!='/': continue 
    if pattern.search( line ):
        sampleName = line
        try:
            castorDir = castorBaseDir + sampleName
        except:
            sys.exit(1)
        localDir = None
        try:
            localDir = os.environ['CMGLOCALBASEDIR']
            localDir += sampleName
        except:
            pass
        allSampleInfo( sampleName, int(options.listLevel) )


    

# allSampleInfo( sampleName, options.listLevel)
