import FWCore.ParameterSet.Config as cms

pfTopProjectionPFTausOnPFJets = cms.EDProducer(
    "PFTopProjectorPFTausOnPFJets",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("No Name"),
    topCollection = cms.InputTag(""),
    bottomCollection = cms.InputTag(""),
)
