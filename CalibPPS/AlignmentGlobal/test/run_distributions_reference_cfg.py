########## Configuration ##########
# if set to True, a file with logs will be produced.
produce_logs = False

# Source max processed events
max_events = 1000000
###################################

import FWCore.ParameterSet.Config as cms

from input_files_reference_cff import input_files

process = cms.Process('referenceDistributions')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("CalibPPS.AlignmentGlobal.ppsAlignmentWorker_cfi")
process.load("DQMServices.Core.DQMStore_cfi")

# Message Logger
if produce_logs:
    process.MessageLogger = cms.Service("MessageLogger",
        destinations = cms.untracked.vstring('run_distributions', 
                                            'cout'
                                            ),
        run_distributions = cms.untracked.PSet(
        	threshold = cms.untracked.string("INFO")
        ),
        cout = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING')
        )
    )
else:
    process.MessageLogger = cms.Service("MessageLogger",
        destinations = cms.untracked.vstring('cout'),
        cout = cms.untracked.PSet(
            threshold = cms.untracked.string('WARNING')
        )
    )

# Source
process.source = cms.Source("PoolSource",
	fileNames = input_files
)
process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(max_events))

# Event Setup
from config_reference_cff import ppsAlignmentConfigESSource
process.ppsAlignmentConfigESSource = ppsAlignmentConfigESSource

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CalibPPS"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CalibPPS"

# Worker config label (for ES product label)
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
