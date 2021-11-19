########## Configuration ##########
# if set to True, a file with logs will be produced.
produce_logs = True

# if set to True, the harvester will produce an extra ROOT file with some debug plots. 
# Works only for one-run input.
harvester_debug = True

# Path for a ROOT file with the histograms.
input_distributions = 'file:dqm_run_distributions_test.root'

# Reference dataset path.
reference_dataset_path = 'DQM_V0001_CalibPPS_R000314273.root'

# If set to True, the results will be also written to an SQLite file.
write_sqlite_results = False

# Output database. Used only if write_sqlite_results is set to True.
output_conditions = 'sqlite_file:alignment_results.db'

# Database tag. Used only if write_sqlite_results is set to True.
output_db_tag = 'CTPPSRPAlignment_real_pcl'
###################################

import sys 
import os
import FWCore.ParameterSet.Config as cms

process = cms.Process('testDistributions')

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load("DQMServices.Core.DQMStore_cfi")
process.load("CalibPPS.AlignmentGlobal.ppsAlignmentHarvester_cfi")

if harvester_debug:
    process.ppsAlignmentHarvester.debug = cms.bool(True)

# Message Logger
if produce_logs:
    process.MessageLogger = cms.Service("MessageLogger",
        destinations = cms.untracked.vstring('run_analysis', 
                                            'cout'
                                            ),
        run_analysis = cms.untracked.PSet(
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

# load DQM framework
process.load("DQMServices.Components.DQMEnvironment_cfi")
process.dqmEnv.subSystemFolder = "CalibPPS"
process.dqmSaver.convention = 'Offline'
process.dqmSaver.workflow = "/CalibPPS/AlignmentGlobal/CMSSW_12_1_0_pre1"
process.dqmSaver.saveByRun = -1
process.dqmSaver.saveAtJobEnd = True
process.dqmSaver.forceRunNumber = 999999

# Source (histograms)
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring(input_distributions),
)

# Event Setup (test)
from config_cff import ppsAlignmentConfigESSource as ppsAlignmentConfigESSourceTest
process.ppsAlignmentConfigESSourceTest = ppsAlignmentConfigESSourceTest

# Event Setup (reference)
from config_reference_cff import ppsAlignmentConfigESSource as ppsAlignmentConfigESSourceReference
ppsAlignmentConfigESSourceReference.matching = cms.PSet(
    reference_dataset = cms.string(reference_dataset_path)
)
process.ppsAlignmentConfigESSourceReference = ppsAlignmentConfigESSourceReference

# SQLite results
if write_sqlite_results:
    process.load("CondCore.CondDB.CondDB_cfi")
    process.CondDB.connect = output_conditions
    process.PoolDBOutputService = cms.Service("PoolDBOutputService",
        process.CondDB,
        timetype = cms.untracked.string('runnumber'),
        toPut = cms.VPSet(cms.PSet(
            record = cms.string('CTPPSRPAlignmentCorrectionsDataRcd'),
            tag = cms.string(output_db_tag)
        ))
    )

    # DB object maker parameters
    process.ppsAlignmentHarvester.write_sqlite_results = cms.bool(True)

process.path = cms.Path(
      process.ppsAlignmentHarvester
)

process.end_path = cms.EndPath(
    process.dqmSaver
)

process.schedule = cms.Schedule(
    process.path,
    process.end_path
)
