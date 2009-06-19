import FWCore.ParameterSet.Config as cms

pfTopProjectionPileUpPFCandidatesOnPFCandidates = cms.EDProducer(
    "PFTopProjectorPileUpPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("No Name"),
    topCollection = cms.InputTag(""),
    bottomCollection = cms.InputTag(""),
)
