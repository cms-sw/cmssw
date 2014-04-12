import FWCore.ParameterSet.Config as cms
process = cms.Process("SKIM")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('JetMETCorrections.Configuration.jecHLTFilters_cfi')
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/mc/Summer09/QCDDiJet_Pt15to20/AODSIM/MC_31X_V3_AODSIM-v1/0046/CEDF080D-CC8F-DE11-A831-001E682F84DE.root')
)

#############   Path       ###########################
process.skimPath = cms.Path(process.HLTDiJetAve15U8E29)

#############   output module ########################
process.compress = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'drop *',
        'keep *_sisCone5CaloJets_*_*',
        'keep *_sisCone7CaloJets_*_*',
        'keep *_kt4CaloJets_*_*',
        'keep *_kt6CaloJets_*_*',
        'keep *_antikt5CaloJets_*_*',
        'keep *_iterativeCone5CaloJets_*_*',  
        'keep *_sisCone5PFJets_*_*',
        'keep *_sisCone7PFJets_*_*',
        'keep *_kt4PFJets_*_*',
        'keep *_kt6PFJets_*_*',
        'keep *_antikt5PFJets_*_*',
        'keep *_iterativeCone5CaloJets_*_*',
        'keep *_iterativeCone5JetExtender_*_*',
        'keep *_sisCone5JetExtender_*_*',
        'keep *_kt4JetExtender_*_*',
        'keep *_TriggerResults_*_*',
        'keep *_hltTriggerSummaryAOD_*_*', 
        'keep *_towerMaker_*_*',
        'keep *_EventAuxilary_*_*',
        'keep *_pixelVertices_*_*',
        'keep *_metHO_*_*',
        'keep *_metNoHF_*_*',
        'keep *_metNoHFHO_*_*', 
        'keep *_met_*_*'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('skimPath')), 
    fileName = cms.untracked.string('JetAOD_HLTDiJetAve15.root')
)

process.p = cms.EndPath(process.compress)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10
