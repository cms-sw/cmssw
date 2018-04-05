import FWCore.ParameterSet.Config as cms

#process = cms.Process('RECODQM')
process = cms.Process('CTPPSDQM')

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
#fileNames=cms.untracked.vstring('file:/afs/cern.ch/user/j/jkaspar/public/run273062_ls0001-2_stream.root')
labelRawDataLikeMC = cms.untracked.bool(False),
fileNames = 
#cms.untracked.vstring('file:/afs/cern.ch/user/p/popov/scratch_bk/data/simevent_CTPPS_DIG_CLU_2_TEST_5000.root')
#cms.untracked.vstring('file:/afs/cern.ch/user/p/popov/public/CTPPS/data/digis_PixelAlive_1294_151_RAW_v2_900p1.root')
#cms.untracked.vstring('file:/afs/cern.ch/user/p/popov/public/CTPPS/data/digis_PixelAlive_P5_2_RAW.root')
cms.untracked.vstring('root://eoscms//eos/cms/store/group/dpg_ctpps/comm_ctpps/digis_PixelAlive_P5_2_RAW.root')

)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")
#process.load("EventFilter.TotemRawToDigi.totemRawToDigi_cff")

# local RP reconstruction chain with standard settings
process.load('Geometry.VeryForwardGeometry.geometryRP_cfi')
process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# CTPPS DQM modules
process.load("DQM.CTPPS.totemDQM_cff")
process.load("DQM.CTPPS.ctppsDQM_cff")

process.options = cms.untracked.PSet(
#    Rethrow = cms.untracked.vstring('ProductNotFound',
    SkipEvent = cms.untracked.vstring('ProductNotFound',
        'TooManyProducts',
        'TooFewProducts')
)

process.path = cms.Path(
  process.ctppsRawToDigi *
  process.recoCTPPS +
  process.ctppsDQM
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
