import FWCore.ParameterSet.Config as cms

from input_files_reference_cff import input_files
from config_reference_cff import ppsAlignmentConfigESSource

process = cms.Process('referenceDistributions')

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

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CalibPPS"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CalibPPS"

process.source = cms.Source("PoolSource",
	fileNames = input_files
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(2000000))

# Event Setup
process.ppsAlignmentConfigESSource = ppsAlignmentConfigESSource

# Worker config label
process.ppsAlignmentWorker.label = cms.string("reference")

process.path = cms.Path(
  	process.ppsAlignmentWorker
)

process.end_path = cms.EndPath(
	process.dqmSaver
)

process.schedule = cms.Schedule(
	process.path,
	process.end_path
)