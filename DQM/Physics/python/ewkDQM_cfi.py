import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK
ewkDQM = cms.EDAnalyzer("EwkDQM",

            elecTriggerPathToPass    = cms.vstring("HLT_Ele", "HLT_DoubleEle", "HLT_DoublePhoton"),
            muonTriggerPathToPass    = cms.vstring("HLT_Mu", "HLT_IsoMu", "HLT_DoubleMu"),
            triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),

            muonCollection           = cms.InputTag("muons"),
            electronCollection       = cms.InputTag("gsfElectrons"),
            PFJetCollection          = cms.InputTag("ak5PFJets"),
            caloMETCollection        = cms.InputTag("corMetGlobalMuons"),
            #genParticleCollection    = cms.InputTag("genParticles")
            EJetMin = cms.untracked.double(15.0)
)
