import FWCore.ParameterSet.Config as cms

greedyMuonPFCandidateFilter = cms.EDFilter(
    "GreedyMuonPFCandidateFilter",
    PFCandidates = cms.InputTag("particleFlow"),
    eOverPMax = cms.double(1.),
    verbose = cms.untracked.bool( True ),
    taggingMode = cms.bool( False ),
    debug = cms.bool( False ),
)
