import FWCore.ParameterSet.Config as cms

jetMETHLTOfflineSource = cms.EDFilter("JetMETHLTOfflineSource",
 
dirname = cms.untracked.string("HLT/JetMET"),
    DQMStore = cms.untracked.bool(True),                      
    verbose = cms.untracked.bool(False),                        
    plotAll = cms.untracked.bool(True),
    plotAllwrtMu = cms.untracked.bool(False), 
    plotEff = cms.untracked.bool(True), 
    pathnameMuon = cms.untracked.vstring("HLT_L1MuOpen"),                      
    pathnameMB = cms.untracked.vstring("HLT_ZeroBias","HLT_MinBiasHcal","HLT_MinBiasEcal"), 
    triggerSummaryLabel = cms.InputTag("hltTriggerSummaryAOD","","HLT"),
    triggerResultsLabel = cms.InputTag("TriggerResults","","HLT"),
    CaloMETCollectionLabel = cms.InputTag("met"),
    CaloJetCollectionLabel = cms.InputTag("iterativeCone5CaloJets"),

    processname = cms.string("HLT")

                                 #-----
                                 
)

