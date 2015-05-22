#!/usr/bin/env python
# Colin
# interface to CRAB, a bit like multicrab

import os, sys, re
from optparse import OptionParser

import CMGTools.Production.eostools as castortools
from CMGTools.Production.addToDatasets import addToDatasets

parser = OptionParser()
parser.usage = """
%prog <dataset name>\nFor a given dataset, prepare a local directory where to run crab, and a destination directory on CASTOR, which complies to the CMG sample organization. Adds the dataset to your local database (~/public/DataSets.txt).

You need to have a valid crab.cfg in the current directory. You can have e.g. crab_data.cfg and crab_MC.cfg. In such a case, just do a symbolic link to use one of them with %prog

Example:

ln -s crab_data.cfg crab.cfg
crabProd.py
cd ./HT/Run2011A-May10ReReco-v1/AOD
crab -create
crab -match
crab -submit

Then, do: 
listSamples.py ^/HT/Run2011A-May10ReReco-v1/AOD$ -l 1
to see where the files are going to appear for this sample (see listSamples.py for more information).

/HT/Run2011A-May10ReReco-v1/AOD, and the files located in the corresponding directory on castor, are called a dataset. More datasets can be created out of this one by running on this dataset using the local batch system (see cmsBatch.py and fwBatch.py). For example, from
/HT/Run2011A-May10ReReco-v1/AOD
one could create:
/HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG
/HT/Run2011A-May10ReReco-v1/AOD/PAT_CMG/RA2
/HT/Run2011A-May10ReReco-v1/AOD/BADEVENTS

It is strongly recommended to use this tool instead of multicrab when running GRID jobs for the CMG group. Otherwise, you will need to manually make sure that the output is stored in the correct output directory on castor, and that your ~/public/DataSets.txt is up-to-date
"""

parser.add_option("-t", "--tier", 
                  dest="tier",
                  help="Tier: extension you can give to specify you are doing a new production. The resulting dataset will be called dataset/tier.",
                  default="")
parser.add_option("-f", "--force", action="store_true",
                  dest="force", 
                  help="Force creation of the destination castor directory. To be used with care, first run without this option",
                  default=False)
parser.add_option("-u", "--user", dest="user", 
                  help="The username to use for the CASTOR location. You must have write permissions",
                  default=os.getlogin())


(options, args) = parser.parse_args()

if len(args)!=1:
    parser.print_help()
    sys.exit(1)

sampleName = args[0]


sampleNameDir = sampleName
if options.tier != "":
    sampleNameDir += "/" + options.tier


# testing that the crab file exists

try:
    oldCrab = open('crab.cfg','r')
except Exception, e:
    print "Cannot find crab.cfg file in current directory. Error was '%s'." % str(e)
    sys.exit(1)

# preparing castor dir -----------------

import castorBaseDir
cdir = castortools.lfnToCastor( castorBaseDir.castorBaseDir(user=options.user) )
cdir += sampleNameDir

if castortools.isCastorFile( cdir ) and not options.force:
    print 'The destination castor directory already exists:'
    print cdir
    print 'Please check. If everything is fine, run again with the -f option.'
    sys.exit(1)

rfmkdir = 'rfmkdir -m 775 -p ' + cdir
print rfmkdir
castortools.createCastorDir(cdir)
castortools.chmod(cdir, '775')

# making local crab directory ---------
ldir = '.' + sampleNameDir

mkdir = 'mkdir -p ' + ldir
print mkdir
os.system( mkdir )

#cpcrab = 'cp crab.cfg %s/crab.cfg' % ldir
#print cpcrab
#os.system( cpcrab )

#prepare the crab file
newCrabPath = '%s/crab.cfg' % ldir
print newCrabPath

newCrab = open(newCrabPath,'w')
newPSet = ""
newJson = ""

patternDataSet = re.compile("\s*datasetpath")
patternRemoteDir = re.compile('\s*user_remote_dir')
patternPSet = re.compile('pset\s*=\s*(.*py)\s*')
patternLumiMask = re.compile('lumi_mask\s*=\s*(\S+)\s*')

pset = None

for line in oldCrab.readlines():
    if patternDataSet.match( line ):
        # removing dataset lines
        continue
    if patternRemoteDir.match( line ):
        # removing remote dir lines 
        continue
    # find and copy parameter set cfg
    match = patternPSet.match( line )
    if match != None:
        pset  = match.group(1)
        newPSet = ldir + "/" + match.group(1)
        os.system('cp %s %s' % (pset, newPSet) )
    # find and copy json file
    match = patternLumiMask.match( line )
    if match != None:
        json  = match.group(1)
        newJson = ldir + "/" + match.group(1)
        os.system('cp %s %s' % (json, newJson) )
   
    newCrab.write( line )

newCrab.write('[CMSSW]\n')
newCrab.write('datasetpath = '+sampleName+'\n')

newCrab.write('[USER]\n')
newCrab.write('user_remote_dir = %s\n' % castortools.castorToLFN(cdir)  )

addToDatasets( sampleNameDir , user = options.user) 

from logger import logger

oldPwd = os.getcwd()
os.chdir(ldir)
logDir = 'Logger'
os.system( 'mkdir ' + logDir )
log = logger( logDir )
log.logCMSSW()
#COLIN not so elegant... but tar is behaving in a strange way.
log.addFile( oldPwd + '/' + pset )
log.addFile( oldPwd + '/' + 'crab.cfg' )
log.stageOut( cdir )

print ''
print 'SUMMARY'
print cdir
print ldir
print newCrabPath
print newPSet
print newJson


