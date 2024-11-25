import FWCore.ParameterSet.Config as cms

process = cms.Process('TOPDQM')

## imports of standard configurations
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

 
from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')

process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '') 

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring())
process.source.skipEvents = cms.untracked.uint32(0)
process.source.fileNames = ['/store/relval/CMSSW_14_2_0_pre3/RelValTTbar_14TeV/MINIAODSIM/PU_140X_mcRun3_2024_realistic_v26_STD_2024_PU-v1/2590000/a102b4b1-a67d-4b75-ae76-438f4dde5deb.root']

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("DQM.Physics.topElectronID_cff")
process.load('Configuration/StandardSequences/Reconstruction_cff')


## output
process.output = cms.OutputModule("PoolOutputModule",
  fileName       = cms.untracked.string('topDQM_production.root'),
  outputCommands = cms.untracked.vstring(
    'drop *_*_*_*',
    'keep *_*_*_TOPDQM',
    'drop *_TriggerResults_*_TOPDQM',
    'drop *_simpleEleId70cIso_*_TOPDQM',
  ),
  splitLevel     = cms.untracked.int32(0),
  dataset = cms.untracked.PSet(
    dataTier   = cms.untracked.string(''),
    filterName = cms.untracked.string('')
  )
)


## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.SingleTopTChannelLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.FwkReport.reportEvery = 100


process.load("DQM.Physics.topSingleLeptonDQM_miniAOD_cfi")
process.load("DQM.Physics.singleTopDQM_miniAOD_cfi")


## path definitions
process.p      = cms.Path(
#    process.simpleEleId70cIso          *
#    process.DiMuonDQM                  +
#    process.DiElectronDQM              +
#    process.ElecMuonDQM                +
    #process.topSingleMuonLooseDQM      +
    process.topSingleMuonMediumDQM_miniAOD  +
    #process.topSingleElectronLooseDQM  +
    process.topSingleElectronMediumDQM_miniAOD +
    process.singleTopMuonMediumDQM_miniAOD     +
    process.singleTopElectronMediumDQM_miniAOD
)
process.endjob = cms.Path(
    process.endOfProcess
)
process.fanout = cms.EndPath(
    process.output
)

## schedule definition
process.schedule = cms.Schedule(
    process.p,
    process.endjob,
    process.fanout
)
