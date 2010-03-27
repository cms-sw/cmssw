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
process.skimPath = cms.Path(process.HLTDiJetAve30U8E29)

#############   output module ########################
process.compress = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
    'drop *',
        #------- CaloJet collections ------
        'keep *_kt4CaloJets_*_*',
        'keep *_kt6CaloJets_*_*',
        'keep *_ak5CaloJets_*_*',
        'keep *_ak7CaloJets_*_*',
        'keep *_iterativeCone5CaloJets_*_*',
        #------- PFJet collections ------  
        'keep *_kt4PFJets_*_*',
        'keep *_kt6PFJets_*_*',
        'keep *_ak5PFJets_*_*',
        'keep *_ak7PFJets_*_*',
        'keep *_iterativeCone5PFJets_*_*',
        #------- JPTJet collections ------
        'keep *_ak5JPTJets_*_*',
        'keep *_iterativeCone5JPTJets_*_*',
        #------- Trigger collections ------
        'keep edmTriggerResults_TriggerResults_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep L1GlobalTriggerObjectMapRecord_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        #------- Tracks collection --------
        'keep *_generalTracks_*_*',
        #------- CaloTower collection -----
        'keep *_towerMaker_*_*',
        #------- Various collections ------
        'keep *_EventAuxilary_*_*',
        'keep *_pixelVertices_*_*',
        'keep *_offlinePrimaryVertices_*_*',
        'keep *_hcalnoise_*_*',
        #------- MET collections ----------
        'keep *_metHO_*_*',
        'keep *_metNoHF_*_*',
        'keep *_metNoHFHO_*_*', 
        'keep *_met_*_*'),
    SelectEvents = cms.untracked.PSet(SelectEvents = cms.vstring('skimPath')), 
    fileName = cms.untracked.string('JetAOD_HLTDiJetAve30.root')
)

process.p = cms.EndPath(process.compress)
#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10
