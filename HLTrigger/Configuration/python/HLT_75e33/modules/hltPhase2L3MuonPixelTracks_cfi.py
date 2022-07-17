import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonPixelTracks = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('hltPhase2L3MuonPixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2L3MuonPixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2L3MuonPixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2L3MuonPixelTracksHitQuadruplets"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('hltPhase2L3MuonPixelTracks')
)
