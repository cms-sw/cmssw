import FWCore.ParameterSet.Config as cms

BadPFMuonSummer16Filter = cms.EDFilter(
    "BadParticleFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   # Collection to test
    muons  = cms.InputTag("muons"),   # Collection to test 
    taggingMode   = cms.bool(False),
    filterType  =cms.string("BadPFMuonSummer16"),
    maxDR         = cms.double(0.001),              # Maximum DR between reco::muon->innerTrack and pfCandidate
    minPtDiffRel = cms.double(0.0),               # lower threshold on difference between pt of reco::muon->innerTrack and pfCandidate
    algo          = cms.int32(14),
    innerTrackRelErr  = cms.double(0.5),               # threshold on inner track pt Error
    minMuonTrackRelErr = cms.double(2.0),          # minimum ptError/pt on muon best track
    minMuonPt       = cms.double(100),               # pt threshold on PF muons 
    segmentCompatibility = cms.double(0.3),        # compatibility between the inner track and the segments in the muon spectrometer
    )
