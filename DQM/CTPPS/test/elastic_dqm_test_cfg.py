import FWCore.ParameterSet.Config as cms

process = cms.Process('RECODQM')

# minimum of logs
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True),
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
    #'/store/data/Run2017B/ZeroBias1/AOD/PromptReco-v2/000/298/570/00000/122D69DA-6A65-E711-BDA8-02163E01375A.root'
    '/store/data/Run2017B/ZeroBias1/AOD/PromptReco-v2/000/298/593/00000/02A251B3-AD65-E711-B189-02163E011E18.root'

    # CTPPS alignment run, 2017, September
    #'/store/data/Run2017E/ZeroBias1/AOD/PromptReco-v1/000/303/649/00000/1270894D-0DA1-E711-BC08-02163E019CBB.root'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# needed for geometry declaration
process.load("Geometry.VeryForwardGeometry.geometryRPFromDD_2017_cfi")

# CTPPS DQM modules
process.load("DQM.CTPPS.elasticPlotDQMSource_cfi")

process.path = cms.Path(
  process.elasticPlotDQMSource
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
