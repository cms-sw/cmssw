import FWCore.ParameterSet.Config as cms

process = cms.Process('RECODQM')

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
  statistics = cms.untracked.vstring(),
  destinations = cms.untracked.vstring('cout'),
  cout = cms.untracked.PSet(
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
    # CTPPS alignment run, 2017, July
    '/store/data/Run2017B/ZeroBias3/AOD/PromptReco-v2/000/298/593/00000/36743C31-A865-E711-8BF2-02163E019D8E.root',
    #'/store/data/Run2017B/ZeroBias6/AOD/PromptReco-v2/000/298/593/00000/6A05E756-A965-E711-9F08-02163E01A760.root',
    #'/store/data/Run2017B/ZeroBias9/AOD/PromptReco-v2/000/298/593/00000/445D9A13-A865-E711-BCFC-02163E01A351.root'

    # CTPPS alignment run, 2017, September
    #'/store/data/Run2017E/ZeroBias1/AOD/PromptReco-v1/000/303/649/00000/1270894D-0DA1-E711-BC08-02163E019CBB.root'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# needed for geometry declaration
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

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
