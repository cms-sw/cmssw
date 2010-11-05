#!/usr/bin/env python
# Colin
# interface to CRAB, a bit like multicrab

import os, sys,  imp, re, pprint, string
from optparse import OptionParser

import castortools

parser = OptionParser()
parser.usage = ""
parser.add_option("-n", "--negate", action="store_true",
                  dest="negate",
                  help="do not proceed",
                  default=False)
parser.add_option("-c", "--castorBaseDir", 
                  dest="castorBaseDir",
                  help="Base castor directory. Subdirectories will be created automatically for each prod",
                  default="/castor/cern.ch/user/c/cbern/cmst3/SusyJetMET")
parser.add_option("-t", "--tier", 
                  dest="tier",
                  help="Tier: extension you can give to specify you are doing a new production",
                  default="")


(options,args) = parser.parse_args()

if len(args)!=1:
    parser.print_help()
    sys.exit(1)

sampleName = args[0]

print 'starting prod for sample:', sampleName

sampleNameDir = sampleName
if options.tier != "":
    sampleNameDir += "/" + options.tier
    

# preparing castor dir -----------------

cdir = options.castorBaseDir 
cdir += sampleNameDir
rfmkdir = 'rfmkdir -p ' + cdir
print rfmkdir
os.system( rfmkdir )
rfchmod = 'rfchmod 775 ' + cdir
print rfchmod
os.system( rfchmod )

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
oldCrab = open('crab.cfg','r')
newPSet = ""
newJson = ""

patternDataSet = re.compile("\s*datasetpath")
patternRemoteDir = re.compile('\s*user_remote_dir')
patternPSet = re.compile('pset=(.*py)\s*')
patternLumiMask = re.compile('lumi_mask\s*=\s*(\S+)\s*')

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

patStripOffCastor = re.compile('/castor/cern.ch/(user/.*)')
match = patStripOffCastor.match( cdir )
newCrab.write('[USER]\n')
newCrab.write('user_remote_dir = %s\n' % match.group(1))

os.system('echo %s >> ~/DataSets.txt' % cdir ) 

print ''
print 'SUMMARY'
print cdir
print ldir
print newCrabPath
print newPSet
print newJson


