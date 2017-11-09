import FWCore.ParameterSet.Config as cms


from Configuration.StandardSequences.Eras import eras

process = cms.Process('DQM',eras.Run2_2017,eras.run2_GEM_2017_MCTest)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

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
process.dqmEnv.subSystemFolder = "GEM"
process.dqmEnv.eventInfoFolder = "EventInfo"
process.dqmSaver.path = ""
process.dqmSaver.tag = "GEM"

process.XMLFromDBSource.label = cms.string("Extended")
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

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
    #'root:///xrootd/store/relval/CMSSW_9_1_1/RelValZMM_14/GEN-SIM-RECO/91X_upgrade2023_realistic_v1_D17-v1/10000/14A91939-1D3F-E711-A09A-0025905A610A.root'
    'file:gem_EDM.root'
  ),
  inputCommands = cms.untracked.vstring(
    'keep *',
    #'keep FEDRawDataCollection_*_*_*'
  )
)

process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

# raw-to-digi conversion
#process.load("EventFilter.CTPPSRawToDigi.ctppsRawToDigi_cff")

# local RP reconstruction chain with standard settings
#process.load("RecoCTPPS.Configuration.recoCTPPS_cff")

# CTPPS DQM modules
process.load("DQM.GEM.GEMDQM_cff")

process.GEMDQMSource.recHitsInputLabel = cms.InputTag("gemRecHits")

process.path = cms.Path(
#  process.ctppsRawToDigi *
#  process.recoCTPPS *
  process.GEMDQM
)

process.end_path = cms.EndPath(
  process.dqmEnv +
  process.dqmSaver
)

process.schedule = cms.Schedule(
  process.path,
  process.end_path
)
