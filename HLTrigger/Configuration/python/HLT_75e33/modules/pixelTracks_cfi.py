import FWCore.ParameterSet.Config as cms

pixelTracks = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('pixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("pixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("pixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("pixelTracksHitSeeds"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('pixelTracks')
)
