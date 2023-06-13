import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3

# needed to have ctppsDQMCalibrationSource properly working
from Configuration.Eras.Modifier_ctpps_cff import ctpps

process = cms.Process('RECODQM', Run3)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cerr'),
  cerr = cms.untracked.PSet(
      threshold = cms.untracked.string('WARNING')
  )
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"

# raw data source, T2 + TrackingStrips test file in run 368593, not 368594 (wrong settings)
process.source = cms.Source("NewEventStreamFileReader",
    fileNames = cms.untracked.vstring(
        '/store/t0streamer/Minidaq/A/000/368/593/run368593_ls0001_streamA_StorageManager.dat',
#        '/store/t0streamer/Minidaq/A/000/368/594/run368594_ls0001_streamA_StorageManager.dat',
    )
)
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(8000)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

#Latest PPS GT candidate
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '130X_dataRun3_Prompt_Candidate_2023_06_06_21_34_08', '')

# local RP reconstruction chain with standard settings
process.load("RecoPPS.Configuration.recoCTPPS_cff")

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")

process.path = cms.Path(
  process.ctppsRawToDigi *
  process.recoCTPPS *
  process.ctppsDQMCalibrationSource *
  process.ctppsDQMCalibrationHarvest
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
