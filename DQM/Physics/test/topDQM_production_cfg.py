import FWCore.ParameterSet.Config as cms

process = cms.Process('TOPDQM')

## imports of standard configurations
#process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('Configuration.StandardSequences.Services_cff')
#process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")

from Configuration.AlCa.GlobalTag import GlobalTag
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic_T15', '')
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

process.source = cms.Source("PoolSource", fileNames = cms.untracked.vstring())
process.source.skipEvents = cms.untracked.uint32(0)

process.source.fileNames = ['root://cms-xrd-global.cern.ch//store/relval/CMSSW_15_0_0_pre1/RelValTTbar_14TeV/GEN-SIM-RECO/PU_142X_mcRun3_2025_realistic_v1_STD_2025_PU-v1/2590000/05925bd0-d592-435e-87c1-fd8f4bdbc726.root']

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

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

## load jet corrections
process.load("JetMETCorrections.Configuration.JetCorrectors_cff")
process.dqmAk4PFCHSL1FastL2L3Corrector = process.ak4PFCHSL1FastL2L3Corrector.clone()
process.jetCorrectorsSeq =  cms.Sequence(process.jetCorrectorsTask)
process.dqmAk4PFCHSL1FastL2L3CorrectorChain = cms.Sequence(process.dqmAk4PFCHSL1FastL2L3Corrector)

## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.SingleTopTChannelLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.MEtoEDMConverter.deleteAfterCopy = cms.untracked.bool(False)  ## line added to avoid crash when changing run number

#process.load("CommonTools.ParticleFlow.EITopPAG_cff")
process.load("DQM.Physics.topSingleLeptonDQM_cfi")
process.load("DQM.Physics.singleTopDQM_cfi")


## path definitions
process.p      = cms.Path(
    
    #process.topSingleMuonLooseDQM      +
    #process.topSingleMuonMediumDQM     +
#    process.EIsequence * 
    process.jetCorrectorsSeq * process.dqmAk4PFCHSL1FastL2L3CorrectorChain *
    process.topSingleMuonMediumDQM      +
    #process.topSingleElectronLooseDQM  +
    #process.ak4PFCHSL1FastL2L3CorrectorChain * 
    process.topSingleElectronMediumDQM  +
    #process.ak4PFCHSL1FastL2L3CorrectorChain * 
    process.singleTopMuonMediumDQM      +
    #process.ak4PFCHSL1FastL2L3CorrectorChain * 
    process.singleTopElectronMediumDQM

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
