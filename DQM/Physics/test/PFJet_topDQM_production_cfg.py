## THIS IS CURRENTLY COMPATIBLE ONLY WITH SINGLE TOP MODULES 
## since the b-tagging algorithms are here re-run with PFJets as input

import FWCore.ParameterSet.Config as cms

process = cms.Process('TOPDQM')

## imports of standard configurations
process.load('DQMOffline.Configuration.DQMOffline_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')



## --------------------------------------------------------------------
## Frontier Conditions: (adjust accordingly!!!)
##
## For CMSSW_3_8_X MC use             ---> 'START38_V12::All'
## For Data (38X re-processing) use   ---> 'GR_R_38X_V13::All'
## For Data (38X prompt reco) use     ---> 'GR10_P_V10::All'
##
## For more details have a look at: WGuideFrontierConditions
## --------------------------------------------------------------------
##process.GlobalTag.globaltag = 'GR_R_42_V14::All' 
#process.GlobalTag.globaltag = 'GR_R_53_V14::All'
process.GlobalTag.globaltag = 'GR_R_52_V7::All'


## input file(s) for testing
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
     #"/store/data/Run2012A/SingleMu/RECO/PromptReco-v1/000/193/116/0AB80D76-FA95-E111-8C46-5404A63886B9.root",
     #"/store/data/Run2012A/SingleMu/RECO/PromptReco-v1/000/193/116/0EDA5C6F-FA95-E111-8681-002481E0E56C.root",      
     #'/store/data/Run2012A/SingleElectron/RECO/PromptReco-v1/000/190/456/1412AF9C-0681-E111-AF6F-003048D2BBF0.root'
     #'/store/relval/CMSSW_5_2_3_patch3/RelValTTbarLepton/GEN-SIM-RECO/START52_V9_special_120410-v1/0122/2C3473C4-1583-E111-8CE8-002618943870.root'
    "/store/data/Run2012B/SingleMu/AOD/PromptReco-v1/000/194/702/A0D074D6-E7A5-E111-A2B7-BCAEC518FF41.root"
     )
)

## number of events
process.maxEvents = cms.untracked.PSet(
  input = cms.untracked.int32(-1)
)

## apply VBTF electronID (needed for the current implementation
## of topSingleElectronDQMLoose and topSingleElectronDQMMedium)
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("DQM.Physics.topElectronID_cff")

# b-tagging
# Re-run b-tagging with PFJets as input
# b-tagging
# Re-run b-tagging with PFJets as input
process.load('Configuration/StandardSequences/Reconstruction_cff')
process.ak4JetTracksAssociatorAtVertex.jets = cms.InputTag("ak4PFJets")
process.p = cms.Path(process.ak4JetTracksAssociatorAtVertex*process.btagging)



#process.topSingleMuonLooseDQM.setup.triggerExtras.src  = cms.InputTag("TriggerResults","","REDIGI42X")
#process.topSingleMuonLooseDQM.preselection.trigger.src = cms.InputTag("TriggerResults","","REDIGI42X")
#process.topSingleMuonLooseDQM.preselection.trigger.select  = cms.vstring(['HLT_Mu15_v2'])
#process.topSingleMuonMediumDQM.preselection.trigger.select = cms.vstring(['HLT_Mu15_v2'])



## output
process.output = cms.OutputModule("PoolOutputModule",
  fileName       = cms.untracked.string('topDQM_production.root'),
  outputCommands = cms.untracked.vstring(
    'drop *_*_*_*',
    'keep *_*_*_TOPDQM',
    'drop *_TriggerResults_*_TOPDQM',
    'drop *_simpleEleId70cIso_*_TOPDQM',
    'drop *_ak4JetTracksAssociatorAtVertex_*_TOPDQM',
    'drop *_btagging_*_TOPDQM',
    'drop *_jetProbabilityBJetTags_*_TOPDQM',
    'drop *_ghostTrackBJetTags_*_TOPDQM',
    'drop *_combinedSecondaryVertexMVABJetTags_*_TOPDQM',
    'drop *_trackCountingHighPurBJetTags_*_TOPDQM',
    'drop *_trackCountingHighEffBJetTags_*_TOPDQM',
    'drop *_simpleSecondaryVertexHighEffBJetTags_*_TOPDQM',
    'drop *_simpleSecondaryVertexHighPurBJetTags_*_TOPDQM',
    'drop *_softElectronByIP3dBJetTags_*_TOPDQM',
    'drop *_softElectronByPtBJetTags_*_TOPDQM',
    'drop *_softMuonBJetTags_*_TOPDQM',
    'drop *_softMuonByIP3dBJetTags_*_TOPDQM',
    'drop *_impactParameterTagInfos_*_TOPDQM',
    'drop *_combinedSecondaryVertexBJetTags_*_TOPDQM',
    'drop *_softMuonByPtBJetTags_*_TOPDQM',
    'drop *_ghostTrackVertexTagInfos_*_TOPDQM',
    'drop *_secondaryVertexTagInfos_*_TOPDQM',
    'drop *_softElectronCands_*_TOPDQM',
    'drop *_softMuonTagInfos_*_TOPDQM',
    'drop *_softElectronTagInfos_*_TOPDQM',
    ),
  splitLevel     = cms.untracked.int32(0),
  dataset = cms.untracked.PSet(
    dataTier   = cms.untracked.string(''),
    filterName = cms.untracked.string('')
  )
)

## load jet corrections
process.load("JetMETCorrections.Configuration.JetCorrectionServicesAllAlgos_cff")
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")
#process.prefer("ak4CaloL2L3")
process.prefer("ak4PFL2L3")


## check the event content
process.content = cms.EDAnalyzer("EventContentAnalyzer")

## configure message logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cerr.threshold = 'INFO'
process.MessageLogger.categories.append('TopSingleLeptonDQM'   )
process.MessageLogger.cerr.TopSingleLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('SingleTopTChannelLeptonDQM'   )
process.MessageLogger.cerr.SingleTopTChannelLeptonDQM    = cms.untracked.PSet(limit = cms.untracked.int32(1))
process.MessageLogger.categories.append('TopDiLeptonOfflineDQM')
process.MessageLogger.cerr.TopDiLeptonOfflineDQM = cms.untracked.PSet(limit = cms.untracked.int32(1))

process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.MEtoEDMConverter.deleteAfterCopy = cms.untracked.bool(False)  ## line added to avoid crash when changing run number


## path definitions
process.p      = cms.Path(
    process.simpleEleId70cIso          *
    process.singleTopMuonMediumDQM     +
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
