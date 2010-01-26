#!/usr/bin/env cmsRun
'''
Copy TaNC training files into a local SQLite file
Adapted from PhysicsTools/MVATrainer/test/testWriteMVAComputerCondDB_cfg.py
Original author: Christopher Saout
Modifications by Evan Friis
'''

import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.TauMVAConfigurations_cfi
import os
from RecoTauTag.TauTagTools.MVASteering_cfi import myTauAlgorithms, GetTrainingFile
import FWCore.ParameterSet.VarParsing as VarParsing

options = VarParsing.VarParsing ('standard')
options.register ('tag',
                  'TauNeuralClassifier', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Name of tag to add")
options.register ('db',
                  'sqlite_file:TancLocal.db', # default value
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "Local database to connect to")

options.parseArguments()

# Make sure we are only dealing w/ one algorithm...
if len(myTauAlgorithms) > 1:
   raise RuntimeError, "ERROR: more than one tau algorithm is defined in MVASteering.py; this feature should be used only for algorithm evaluation.  \
         Please modify it so that it only includeds the algorithm on which the TaNC is to be used."

algorithm = myTauAlgorithms[0]
myconnect   = cms.string(options.db)  #or frontier, etc
mytag       = cms.string(options.tag)
mytimetype  = cms.untracked.string('runnumber')
print ""
print "***************************************************"
print "******  Upload Tau Neural Classifier to DB   ******"
print "***************************************************"
print "*  Using the %s algorithm                         " % algorithm
print "*  DB tag:       %s                               " % mytag.value()
print "*  Database:     %s                               " % myconnect.value()
print "*  Timetype:     %s                               " % mytimetype.value()
print "* ----------------------------------------------- "

# Unpack the TaNC neural nets into a parameter set
tempPSet   = cms.PSet()
toCopyList = cms.vstring()

for aNeuralNet in RecoTauTag.TauTagTools.TauMVAConfigurations_cfi.TaNC.value():
   # Get the name of this neural net
   neuralNetName = aNeuralNet.computerName.value()
   # Make sure we have the .mva training done
   mvaFileLocation = GetTrainingFile(neuralNetName, algorithm)
   if not os.path.exists(mvaFileLocation):
      raise IOError, "Expected trained .mva file at %s, it doesn't exist!" % mvaFileLocation
   # god bless you, python
   tempPSet.__setattr__(aNeuralNet.computerName.value(), cms.string(mvaFileLocation))
   toCopyList.append(neuralNetName)
   print "* %-20s %-20s      " % (neuralNetName, mvaFileLocation )

process = cms.Process("TaNCCondUpload")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(1) )

process.MVAComputerESSource = cms.ESSource("TauMVAComputerESSource",
      tempPSet  # defined above, maps the Tanc NN names to their trained MVA weights files
)

process.MVAComputerSave = cms.EDAnalyzer("TauMVATrainerSave",
	toPut = cms.vstring(),
        #list of labels to add into the tag given in the PoolDBOutputService
	#toCopy = cms.vstring('ZTauTauTraining', 'ZTauTauTrainingCopy2')
	toCopy = toCopyList
)

process.PoolDBOutputService = cms.Service("PoolDBOutputService",
	BlobStreamerName = cms.untracked.string('TBufferBlobStreamingService'),
	DBParameters = cms.PSet( messageLevel = cms.untracked.int32(4) ),
	timetype = mytimetype,
	connect = myconnect,
	toPut = cms.VPSet(cms.PSet(
		record = cms.string('TauTagMVAComputerRcd'),
		tag = mytag
	))
)

process.outpath = cms.EndPath(process.MVAComputerSave)
