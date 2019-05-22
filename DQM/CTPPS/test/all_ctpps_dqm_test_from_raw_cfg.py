import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Modifier_ctpps_2016_cff import ctpps_2016
process = cms.Process('ctppsDQMfromRAW', ctpps_2016)

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
    # reference for 2016, pre-TS2 (fill 5029)
    #"/store/data/Run2016B/ZeroBias/RAW/v2/000/275/371/00000/FA67145B-8836-E611-8560-02163E012627.root",
    #"/store/data/Run2016B/ZeroBias/RAW/v2/000/275/371/00000/EAD70032-8836-E611-8C11-02163E014154.root",
    "/store/data/Run2016B/DoubleEG/RAW/v2/000/275/371/00000/FE9F0F13-9436-E611-8F39-02163E012B47.root",  # temporarily use a staged file from a different stream

    # referece for 2016, post-TS2 (fill 5424)
    #"/store/data/Run2016H/ZeroBias/RAW/v1/000/283/453/00000/463B84C2-C098-E611-8BC4-FA163E3201B4.root",
    #"/store/data/Run2016H/ZeroBias/RAW/v1/000/283/453/00000/3204EE5B-C298-E611-BC39-02163E01448F.root",
    "/store/data/Run2016H/SingleMuon/RAW/v1/000/283/453/00000/FE53CBFE-CB98-E611-A106-FA163E04425A.root",  # temporarily use a staged file from a different stream

    # referece for 2017, pre-TS2 (fill 6089)
    "/store/data/Run2017C/ZeroBias/RAW/v1/000/301/283/00000/8ED63519-2282-E711-9073-02163E01A3C6.root",
    #"/store/data/Run2017C/ZeroBias/RAW/v1/000/301/283/00000/D4508469-2282-E711-82A9-02163E01A31A.root",

    # referece for 2017, post-TS2 (fill 6300)
    "/store/data/Run2017F/ZeroBias/RAW/v1/000/305/081/00000/001D08EE-C4B1-E711-B92D-02163E013864.root",
    #"/store/data/Run2017F/ZeroBias/RAW/v1/000/305/081/00000/44B0284D-C3B1-E711-BECF-02163E014357.root",

    # referece for 2018 (fill 7006)
    "/store/data/Run2018D/ZeroBias/RAW/v1/000/320/688/00000/601A721D-AD95-E811-B21A-FA163E28A50A.root",
    #"/store/data/Run2018D/ZeroBias/RAW/v1/000/320/688/00000/EE97DF44-AD95-E811-A444-02163E019FF7.root",
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
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# CTPPS DQM modules
process.load("DQM.CTPPS.ctppsDQM_cff")

process.path = cms.Path(
  process.ctppsRawToDigi *
  process.recoCTPPS *
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
