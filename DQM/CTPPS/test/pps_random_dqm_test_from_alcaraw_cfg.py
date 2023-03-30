import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_ctpps_cff import ctpps
process = cms.Process('ctppsRandomDQMfromRAW', ctpps)

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cerr'),
  cerr = cms.untracked.PSet(
      threshold = cms.untracked.string('WARNING')
  )
)

# load DQM framework
process.load("DQM.Integration.config.environment_cfi")
process.dqmEnv.subSystemFolder = "CTPPS"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "CTPPS"

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    '/store/group/dpg_ctpps/comm_ctpps/PixelRandomTrigger2023/outputExpressPPSRandom.root'
  ),

  inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_*_*_*'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# global tag - conditions for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# raw-to-digi conversion
from EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff import ctppsPixelDigis as _ctppsPixelDigis
process.ctppsPixelDigisAlCaRecoProducer = _ctppsPixelDigis.clone(inputLabel = 'hltPPSCalibrationRaw')

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")

process.path = cms.Path(
  process.ctppsPixelDigisAlCaRecoProducer *
  process.ctppsDQMRandomSource *
  process.ctppsDQMRandomHarvest
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
