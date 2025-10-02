import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpakaPhase2",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    trackSrc = cms.InputTag("hltPhase2PixelTracksSoA"),
    outerTrackerRecHitSrc = cms.InputTag(""),
    useOTExtension = cms.bool(False),
    requireQuadsFromConsecutiveLayers = cms.bool(True)
)

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
_hltPhase2PixelTracksCAExtensionSelectionHighPurity = cms.EDProducer("TrackCollectionFilterCloner",
    copyExtras = cms.untracked.bool(True),
    copyTrajectories = cms.untracked.bool(False),
    minQuality = cms.string('highPurity'),
    originalMVAVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","MVAValues"),
    originalQualVals = cms.InputTag("hltPhase2PixelTracksCutClassifier","QualityMasks"),
    originalSource = cms.InputTag("hltPhase2PixelTracksCAExtension")
)
phase2CAExtension.toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracksCAExtensionSelectionHighPurity)

from Configuration.ProcessModifiers.phase2LegacyPixelTracks_cff import phase2LegacyPixelTracks
_hltPhase2PixelTracksLegacy = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('pixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2PixelTracksHitSeeds"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('hltPhase2PixelTracks')
)
phase2LegacyPixelTracks.toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracksLegacy)