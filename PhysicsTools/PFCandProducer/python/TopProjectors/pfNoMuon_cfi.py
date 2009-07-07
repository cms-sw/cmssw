import FWCore.ParameterSet.Config as cms

pfNoMuon = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noMuon"),
    topCollection = cms.InputTag("pfIsolatedMuons"),
    bottomCollection = cms.InputTag("pfNoPileUp"),
)
