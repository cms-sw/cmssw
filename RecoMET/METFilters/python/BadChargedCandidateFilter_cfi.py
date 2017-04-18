import FWCore.ParameterSet.Config as cms

BadChargedCandidateFilter = cms.EDFilter(
    "BadParticleFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    muons  = cms.InputTag("muons"),   # Collection to test
    taggingMode   = cms.bool(False),
    filterType  =cms.string("BadChargedCandidate"),
    maxDR         = cms.double(0.00001),              # Maximum DR between reco::muon->innerTrack and pfCandidate 
    minPtDiffRel = cms.double(0.00001),               # lower threshold on difference between pt of reco::muon->innerTrack and pfCandidate
    minMuonTrackRelErr = cms.double(2.0),          # minimum ptError/pt on muon best track
    innerTrackRelErr   = cms.double(1.0),          # minimum relPtErr on innerTrack
    minMuonPt     = cms.double(100.0),               # minimum muon pt 
    segmentCompatibility = cms.double(0.3),        # compatibility between the inner track and the segments in the muon spectrometer
)
