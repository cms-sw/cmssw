import FWCore.ParameterSet.Config as cms
process = cms.Process("SKIM")
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('JetMETCorrections.Configuration.jecHLTFilters_cfi')
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1000)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
'/store/relval/CMSSW_3_5_5/RelValQCD_FlatPt_15_3000/GEN-SIM-RECO/MC_3XY_V25-v1/0007/E0707AC5-0F38-DF11-B1FD-0026189437E8.root'
)
)

#############   Path       ###########################
process.skimPath = cms.Path(process.HLTDiJetAve30U1E31)

#############   output module ########################
process.compress = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
    'drop *',
        #------- CaloJet collections ------
        'keep recoCaloJets_kt4CaloJets_*_*',
        'keep recoCaloJets_kt6CaloJets_*_*',
        'keep recoCaloJets_ak5CaloJets_*_*',
        'keep recoCaloJets_ak7CaloJets_*_*',
        'keep recoCaloJets_iterativeCone5CaloJets_*_*',
        #------- CaloJet ID ---------------
        'keep *_kt4JetID_*_*',
        'keep *_kt6JetID_*_*',
        'keep *_ak5JetID_*_*',
        'keep *_ak7JetID_*_*',
        'keep *_ic5JetID_*_*', 
        #------- PFJet collections ------  
        'keep recoPFJets_kt4PFJets_*_*',
        'keep recoPFJets_kt6PFJets_*_*',
        'keep recoPFJets_ak5PFJets_*_*',
        'keep recoPFJets_ak7PFJets_*_*',
        'keep recoPFJets_iterativeCone5PFJets_*_*',
        #------- JPTJet collections ------
        #'keep *_ak5JPTJets_*_*',
        #'keep *_iterativeCone5JPTJets_*_*',
        #------- Trigger collections ------
        'keep edmTriggerResults_TriggerResults_*_*',
        'keep *_hltTriggerSummaryAOD_*_*',
        'keep L1GlobalTriggerObjectMapRecord_*_*_*',
        'keep L1GlobalTriggerReadoutRecord_*_*_*',
        #------- Tracks collection --------
        'keep recoTracks_generalTracks_*_*',
        #------- CaloTower collection -----
        'keep *_towerMaker_*_*',
        #------- Various collections ------
        'keep *_EventAuxilary_*_*',
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
