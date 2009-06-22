import FWCore.ParameterSet.Config as cms

noMuon = cms.EDProducer(
    "TPPFCandidatesOnPFCandidates",
    verbose = cms.untracked.bool( False ),
    name = cms.untracked.string("noMuon"),
    topCollection = cms.InputTag("pfMuons"),
    bottomCollection = cms.InputTag("noPileUp"),
)
