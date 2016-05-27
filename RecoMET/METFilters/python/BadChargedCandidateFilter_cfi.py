import FWCore.ParameterSet.Config as cms

BadChargedCandidateFilter = cms.EDFilter(
    "BadChargedCandidateFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    muons  = cms.InputTag("muons"),   # Collection to test
    taggingMode   = cms.bool(False),
    debug         = cms.bool(False),
    maxDR         = cms.double(0.01),               # Maximum DR between reco::muon->innerTrack and pfCandidate 
    minPtErrorRel = cms.double(-0.5),               # lower threshold on difference between pt of reco::muon->innerTrack and pfCandidate
    minMuonPt     = cms.double(20),                 # minimum muon pt 
)
