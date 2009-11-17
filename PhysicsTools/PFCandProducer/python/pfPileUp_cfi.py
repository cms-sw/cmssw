import FWCore.ParameterSet.Config as cms


pfPileUp = cms.EDProducer(
    "PFPileUp",
    PFCandidates = cms.InputTag("particleFlow"),
    Vertices = cms.InputTag("offlinePrimaryVerticesWithBS"),
    # pile-up identification now disabled by default. To be studied
    Enable = cms.bool(False),
    verbose = cms.untracked.bool(False)
    )
