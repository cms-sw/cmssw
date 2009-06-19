import FWCore.ParameterSet.Config as cms

pfTopProjectionIsolatedPFCandidatesOnPFCandidates = cms.EDProducer(
    "PFTopProjectorIsolatedPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("No Name"),
    topCollection = cms.InputTag(""),
    bottomCollection = cms.InputTag(""),
)
