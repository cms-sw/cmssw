import FWCore.ParameterSet.Config as cms

process = cms.Process("TestZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    'file:3ADEF5CA-1511-E011-ADD2-0017A4770010.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START39_V8::All')
process.load("Configuration.StandardSequences.MagneticField_cff")

############
## To run on data or without MC truth uncomment the following
#process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPaths_cff")
############

# Output module configuration
process.load("ElectroWeakAnalysis.Skimming.zMuMuSubskimOutputModule_cfi")

############
# To run MC truth uncomment the following
process.load("ElectroWeakAnalysis.Skimming.zMuMu_SubskimPathsWithMCTruth_cff")
process.zMuMuSubskimOutputModule.outputCommands.extend(process.mcEventContent.outputCommands)
############

process.zMuMuSubskimOutputModule.fileName = 'file:testZMuMuSubskim.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)

### Here set the HLT Path for trigger filtering and matching
process.dimuonsHLTFilter.HLTPaths = ["HLT_Mu9", "HLT_Mu11", "HLT_Mu15_v1"]
process.muonTriggerMatchHLTMuons.pathNames = cms.vstring( 'HLT_Mu15_v1' )
process.userDataMuons.hltPath = cms.string("HLT_Mu15_v1")
process.userDataDimuons.hltPath = cms.string("HLT_Mu15_v1")
process.userDataDimuonsOneTrack.hltPath = cms.string("HLT_Mu15_v1")
############

### 3_9_X reprocessed MC: to process REDIGI HLT tables uncomment the following
#process.dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","REDIGI39X")
#process.patTrigger.processName = "REDIGI39X"
#process.patTriggerEvent.processName = "REDIGI39X"
#process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI39X" )
#process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI39X" )




