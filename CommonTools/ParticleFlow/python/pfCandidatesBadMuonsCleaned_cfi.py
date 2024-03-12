import FWCore.ParameterSet.Config as cms

pfCandidatesBadMuonsCleaned = cms.EDProducer("PFCandidateMuonUntagger",
    pfcandidates = cms.InputTag("particleFlow"),
    oldToNewMuons = cms.InputTag("muonsCleaned","oldToNew"), 
)
# foo bar baz
# aNsZfxxxo8xcd
# 62NhqIQ39Nx7G
