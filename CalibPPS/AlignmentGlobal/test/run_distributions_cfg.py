import FWCore.ParameterSet.Config as cms

from input_files_cff import input_files
from config_cff import ppsAlignmentConfigESSource

process = cms.Process('testDistributions')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("CalibPPS.AlignmentGlobal.ppsAlignmentWorker_cfi")
process.load("DQMServices.Core.DQMStore_cfi")

# Message Logger
process.MessageLogger = cms.Service("MessageLogger",
	destinations = cms.untracked.vstring(# 'run_distributions', 
	                                     'cout'
	                                    ),
	# run_distributions = cms.untracked.PSet(
	# 	threshold = cms.untracked.string("INFO")
	# ),
	cout = cms.untracked.PSet(
		threshold = cms.untracked.string('WARNING')
	)
)

process.source = cms.Source("PoolSource",
	fileNames = input_files
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))

# Event Setup
process.ppsAlignmentConfigESSource = ppsAlignmentConfigESSource

process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
	fileName = cms.untracked.string("dqm_run_distributions_test.root")
)

process.path = cms.Path(
  	process.ppsAlignmentWorker
)

process.end_path = cms.EndPath(
	process.dqmOutput
)

process.schedule = cms.Schedule(
	process.path,
	process.end_path
)