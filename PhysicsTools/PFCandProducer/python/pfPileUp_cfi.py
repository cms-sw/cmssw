import FWCore.ParameterSet.Config as cms


pfPileUp = cms.EDProducer(
    "PFPileUp",
    PFCandidates = cms.InputTag("particleFlow"),
    Vertices = cms.InputTag("offlinePrimaryVerticesWithBS"),
    verbose = cms.untracked.bool(False)
    )
