import FWCore.ParameterSet.Config as cms

process = cms.Process("TestZMuMuSubskim")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.options   = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

# source
process.source = cms.Source("PoolSource", 
     fileNames = cms.untracked.vstring(
    '/store/relval/CMSSW_3_8_5/RelValZMM/GEN-SIM-RECO/START38_V12-v1/0041/1C1BBE0B-D2D2-DF11-BDA3-002618943852.root'
    )
)
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )

process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = cms.string('START38_V12::All')
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

process.zMuMuSubskimOutputModule.fileName = 'file:/tmp/fabozzi/testZMuMuSubskim.root'

process.outpath = cms.EndPath(process.zMuMuSubskimOutputModule)

### Here set the HLT Path for trigger matching
process.muonTriggerMatchHLTMuons.pathNames = cms.vstring( 'HLT_Mu11' )
process.userDataMuons.hltPath = cms.string("HLT_Mu11")
process.userDataDimuons.hltPath = cms.string("HLT_Mu11")
process.userDataDimuonsOneTrack.hltPath = cms.string("HLT_Mu11")
############

### 3_6_X reprocessed MC: to process REDIGI HLT tables uncomment the following
#process.dimuonsHLTFilter.TriggerResultsTag = cms.InputTag("TriggerResults","","REDIGI36X")
#process.patTrigger.processName = "REDIGI36X"
#process.patTriggerEvent.processName = "REDIGI36X"
#process.patTrigger.triggerResults = cms.InputTag( "TriggerResults::REDIGI36X" )
#process.patTrigger.triggerEvent = cms.InputTag( "hltTriggerSummaryAOD::REDIGI36X" )




