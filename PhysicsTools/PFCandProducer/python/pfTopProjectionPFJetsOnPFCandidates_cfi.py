import FWCore.ParameterSet.Config as cms

pfTopProjectionPFJetsOnPFCandidates = cms.EDProducer(
    "PFTopProjectorPFJetsOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("No Name"),
    topCollection = cms.InputTag(""),
    bottomCollection = cms.InputTag(""),
)
