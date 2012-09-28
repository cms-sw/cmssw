import FWCore.ParameterSet.Config as cms

pfNoPileUp = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    enable =  cms.bool( True ),
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("pileUpOnPFCandidates"),
    topCollection = cms.InputTag("pfPileUp"),
    bottomCollection = cms.InputTag("particleFlowTmpPtrs"),
)
