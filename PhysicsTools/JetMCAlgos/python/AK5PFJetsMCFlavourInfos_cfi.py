import FWCore.ParameterSet.Config as cms

jetFlavourInfos = cms.EDProducer("JetFlavourClustering",
    jets           = cms.InputTag("ak5PFJets"),
    bHadrons       = cms.InputTag("selectedHadronsAndPartons","bHadrons"),
    cHadrons       = cms.InputTag("selectedHadronsAndPartons","cHadrons"),
    partons        = cms.InputTag("selectedHadronsAndPartons","partons"),
    jetAlgorithm   = cms.string("AntiKt"),
    rParam         = cms.double(0.5),
    ghostRescaling = cms.double(1e-18)
)
