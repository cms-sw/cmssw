import FWCore.ParameterSet.Config as cms

pfCandidatesBadMuonsCleaned = cms.EDProducer("PFCandidateMuonUntagger",
    pfcandidates = cms.InputTag("particleFlow"),
    oldToNewMuons = cms.InputTag("muonsCleaned","oldToNew"), 
)
