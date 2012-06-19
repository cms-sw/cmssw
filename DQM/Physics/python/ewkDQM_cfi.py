import FWCore.ParameterSet.Config as cms

# DQM monitor module for EWK
ewkDQM = cms.EDAnalyzer("EwkDQM",
            elecTriggerPathToPass    = cms.string("HLT_Ele"),
            muonTriggerPathToPass    = cms.string("HLT_Mu"),
            #eleTrigPathNames = cms.untracked.vstring("HLT_Ele","HLT_DoubleEle"),
            #muTrigPathNames = cms.untracked.vstring("HLT_Mu","HLT_DoubleMu","HLT_IsoMu"),
            triggerResultsCollection = cms.InputTag("TriggerResults","","HLT"),
            muonCollection           = cms.InputTag("muons"),
            electronCollection       = cms.InputTag("gsfElectrons"),
            PFJetCollection          = cms.InputTag("ak5PFJets"),
            caloMETCollection        = cms.InputTag("corMetGlobalMuons"),
            #genParticleCollection    = cms.InputTag("genParticles")
#           caloJetCollection        = cms.InputTag("L2L3CorJetSC5Calo"),
            EJetMin = cms.untracked.double(15.0)
)
