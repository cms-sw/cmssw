import FWCore.ParameterSet.Config as cms

hltPhase2L3FromL1TkMuonPixelTracks = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('hltPixelTracksCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2L3MuonPixelTracksFilter"),
    Fitter = cms.InputTag("hltPhase2L3MuonPixelTracksFitter"),
    SeedingHitSets = cms.InputTag("hltPhase2L3FromL1TkMuonPixelTracksHitQuadruplets"),
    passLabel = cms.string('')
)
