import FWCore.ParameterSet.Config as cms

pfNoTau = cms.EDProducer(
    "TPPFTausOnPFJets",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noTau"),
    topCollection = cms.InputTag("pfTaus"),
    bottomCollection = cms.InputTag("pfJets"),
)
