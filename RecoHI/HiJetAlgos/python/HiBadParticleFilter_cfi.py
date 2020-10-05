import FWCore.ParameterSet.Config as cms

filteredParticleFlow = cms.EDFilter(
    "HiBadParticleFilter",
    PFCandidates  = cms.InputTag("particleFlow"),   
    offlinePV  = cms.InputTag("offlinePrimaryVertices"),   
    taggingMode   = cms.bool(False),
    verbose   = cms.bool(False),
    minMuonTrackRelErr = cms.double(2.0),          # minimum ptError/pt on muon best track
    minMuonPt     = cms.double(20.0),               # minimum muon pt 
    minChargedHadronPt = cms.double(20.0),
    minMuonTrackRelPtErr = cms.double(2.),
    maxSigLoose = cms.double(100.),
    maxSigTight = cms.double(10.),
    minCaloCompatibility = cms.double(0.35),
    minTrackNHits = cms.uint32(10),
    minPixelNHits = cms.uint32(3)
)
