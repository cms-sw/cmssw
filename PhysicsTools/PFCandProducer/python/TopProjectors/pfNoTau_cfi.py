import FWCore.ParameterSet.Config as cms

pfNoTau = cms.EDProducer(
    "TPPFTausOnPFJets",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noTau"),
    topCollection = cms.InputTag("allLayer0Taus"),
    bottomCollection = cms.InputTag("pfJets"),
)
