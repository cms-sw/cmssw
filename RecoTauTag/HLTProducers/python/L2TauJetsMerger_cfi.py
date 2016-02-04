import FWCore.ParameterSet.Config as cms

l2TauJetsMerger = cms.EDProducer("L2TauJetsMerger",
    EtMin = cms.double(15.0),
    JetSrc = cms.VInputTag(cms.InputTag("icone5Tau1"), cms.InputTag("icone5Tau2"), cms.InputTag("icone5Tau3"), cms.InputTag("icone5Tau4"))
)


