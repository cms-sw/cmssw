import FWCore.ParameterSet.Config as cms

BadChargedCandidateFilter = cms.EDFilter(
    "BadChargedCandidateFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    muons  = cms.InputTag("muons"),   # Collection to test
    taggingMode   = cms.bool(False),
    debug         = cms.bool(False),
    maxDR         = cms.double(0.001),              # Maximum DR between reco::muon->innerTrack and pfCandidate 
    minPtDiffRel = cms.double(-0.5),               # lower threshold on difference between pt of reco::muon->innerTrack and pfCandidate
                                                   # computed as (pfcand.pt - muon.track.pt)/(0.5*(pfcand.pt + muon.track.pt))
    minMuonTrackRelErr = cms.double(2.0),          # minimum ptError/pt on muon best track
    minMuonPt     = cms.double(100),               # minimum muon pt 
    segmentCompatibility = cms.double(0.3),        # compatibility between the inner track and the segments in the muon spectrometer

)
