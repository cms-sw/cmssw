import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('ctppsDQMfromAOD', eras.ctpps_2016)

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

# import of standard configurations for pixel
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_hlt_relval', '')

# raw data source
process.source = cms.Source("PoolSource",
  fileNames = cms.untracked.vstring(
    # run 314255, alignment run April 2018
    '/store/data/Commissioning2018/ZeroBias1/AOD/PromptReco-v1/000/314/255/00000/0098845D-D642-E811-8E30-FA163EE31D2A.root'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# geometry definition
process.load("Geometry.VeryForwardGeometry.geometryRPFromDB_cfi")

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")

process.path = cms.Path(
  process.ctppsDQMElastic
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
