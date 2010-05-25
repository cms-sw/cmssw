import FWCore.ParameterSet.Config as cms

l2TauJetsProvider = cms.EDProducer("L2TauJetsProvider",
    L1Particles = cms.InputTag("l1extraParticles","Tau"),
    L1TauTrigger = cms.InputTag("DummyHLTL1SeedFilter"),
    EtMin = cms.double(15.0),
    JetSrc = cms.VInputTag(cms.InputTag("icone5Tau1"), cms.InputTag("icone5Tau2"), cms.InputTag("icone5Tau3"), cms.InputTag("icone5Tau4"))
)


