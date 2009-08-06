import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK
ewkDQM = cms.EDAnalyzer("EwkDQM",
            elecTriggerPathToPass    = cms.string("HLT_Ele10_LW_L1R"),
            muonTriggerPathToPass    = cms.string("HLT_Mu9"),
            triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT8E29"),
            muonCollection           = cms.InputTag("muons"),
            electronCollection       = cms.InputTag("gsfElectrons"),
            caloJetCollection        = cms.InputTag("sisCone5CaloJets"),
            caloMETCollection        = cms.InputTag("corMetGlobalMuons"),
            genParticleCollection    = cms.InputTag("genParticles")
#           caloJetCollection        = cms.InputTag("L2L3CorJetSC5Calo"),
)
