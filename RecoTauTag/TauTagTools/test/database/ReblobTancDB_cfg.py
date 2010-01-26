#!/usr/bin/env cmsRun

import FWCore.ParameterSet.Config as cms
import RecoTauTag.TauTagTools.TauMVAConfigurations_cfi
import os

from RecoTauTag.TauTagTools.MVASteering_cfi import *

# Copy TaNC training files into a local SQLite file
# Adapted from PhysicsTools/MVATrainer/test/testWriteMVAComputerCondDB_cfg.py
# Original author: Christopher Saout
# Modifications by Evan Friis

# Make sure we are only dealing w/ one algorithm...
if len(myTauAlgorithms) > 1:
   raise RuntimeError, "ERROR: more than one tau algorithm is defined in MVASteering.py; this feature should be used only for algorithm evaluation.  \
         Please modify it so that it only includeds the algorithm on which the TaNC is to be used."

algorithm = myTauAlgorithms[0]
myconnect   = cms.string('sqlite_file:TancLocal.db')  #or frontier, etc
mytag       = cms.string('TauNeuralClassifier_v2')
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


process = cms.Process("TaNCCondUpload")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(	input = cms.untracked.int32(1) )

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = 'STARTUP31X_V7::All'

toCopyList = cms.vstring()
for aNeuralNet in RecoTauTag.TauTagTools.TauMVAConfigurations_cfi.TaNC.value():
   neuralNetName = aNeuralNet.computerName.value()
   toCopyList.append(neuralNetName)

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
