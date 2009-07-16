import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK
ewkDQM = cms.EDAnalyzer("EwkDQM",
                            triggerPathToPass        = cms.string("HLT_Ele15_LW_L1R"),
                            triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),
                            muonCollection           = cms.InputTag("muons"),
                            electronCollection       = cms.InputTag("gsfElectrons"),
                            caloJetCollection        = cms.InputTag("sisCone5CaloJets"),
                            caloMETCollection        = cms.InputTag("met"),
                            genParticleCollection    = cms.InputTag("genParticles")
#                           caloJetCollection        = cms.InputTag("L2L3CorJetSC5Calo"),
)
