import FWCore.ParameterSet.Config as cms

pfNoPileUp = cms.EDProducer(
    "TPPileUpPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("pileUpOnPFCandidates"),
    topCollection = cms.InputTag("pfPileUp"),
    bottomCollection = cms.InputTag("particleFlow"),
)
