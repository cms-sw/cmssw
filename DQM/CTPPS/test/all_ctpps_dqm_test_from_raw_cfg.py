import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras
process = cms.Process('ctppsDQMfromRAW', eras.ctpps_2016)

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
    # run 274199, fill 4961, 29 May 2016 (before TS2)
    #'/store/data/Run2016B/DoubleEG/RAW/v2/000/274/199/00000/04985451-9B26-E611-BEB9-02163E013859.root',
    #'root://eostotem.cern.ch//eos/totem/user/j/jkaspar/04C8034A-9626-E611-9B6E-02163E011F93.root'

    # run 283877, fill 5442, 23 Oct 2016 (after TS2)
    #'/store/data/Run2016H/HLTPhysics/RAW/v1/000/283/877/00000/F28F8896-999B-E611-93D8-02163E013706.root',

    # test file for 2017 mapping (vertical RPs only)
    #'root://eostotem.cern.ch//eos/totem/data/ctpps/run290874.root'

    # 900GeV test, 8 May 2018
    '/store/data/Run2018A/Totem1/RAW/v1/000/315/956/00000/B2AB3BFA-8D53-E811-ACFA-FA163E63AE40.root'
  ),
  inputCommands = cms.untracked.vstring(
    'drop *',
    'keep FEDRawDataCollection_*_*_*'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(8000)
)

# global tag - conditions for P5 cluster
process.load("DQM.Integration.config.FrontierCondition_GT_cfi")

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load("RecoCTPPS.Configuration.recoCTPPS_DD_cff")

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
