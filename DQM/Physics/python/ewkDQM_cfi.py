import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK
ewkDQM = cms.EDAnalyzer("EwkDQM",

            elecTriggerPathToPass    = cms.vstring("HLT_Ele", "HLT_DoubleEle", "HLT_DoublePhoton"),
            muonTriggerPathToPass    = cms.vstring("HLT_Mu", "HLT_IsoMu", "HLT_DoubleMu"),
            triggerResultsCollection = cms.InputTag("TriggerResults", "", "HLT"),

            muonCollection           = cms.InputTag("muons"),
            electronCollection       = cms.InputTag("gedGsfElectrons"),
            PFJetCollection          = cms.InputTag("ak4PFJets"),
            caloMETCollection        = cms.InputTag("caloMetM"),
            vertexCollection         = cms.InputTag("offlinePrimaryVertices"),
            #genParticleCollection    = cms.InputTag("genParticles")
            EJetMin = cms.untracked.double(15.0)
)
