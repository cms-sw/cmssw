import FWCore.ParameterSet.Config as cms

pfTopProjectionPFCandidatesOnPFCandidates = cms.EDProducer(
    "PFTopProjectorPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("No Name"),
    topCollection = cms.InputTag(""),
    bottomCollection = cms.InputTag(""),
)
