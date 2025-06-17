import FWCore.ParameterSet.Config as cms

hltPhase2PixelTracks = cms.EDProducer("PixelTrackProducer",
    Cleaner = cms.string('pixelTrackCleanerBySharedHits'),
    Filter = cms.InputTag("hltPhase2PixelTrackFilterByKinematics"),
    Fitter = cms.InputTag("hltPhase2PixelFitterByHelixProjections"),
    SeedingHitSets = cms.InputTag("hltPhase2PixelTracksHitSeeds"),
    mightGet = cms.optional.untracked.vstring,
    passLabel = cms.string('hltPhase2PixelTracks')
)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
_hltPhase2PixelTracks = cms.EDProducer("PixelTrackProducerFromSoAAlpakaPhase2",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    trackSrc = cms.InputTag("hltPhase2PixelTracksSoA"),
    outerTrackerRecHitSrc = cms.InputTag(""),
    useOTExtension = cms.bool(False)
)
alpaka.toReplaceWith(hltPhase2PixelTracks, _hltPhase2PixelTracks)

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toModify(hltPhase2PixelTracks,
    outerTrackerRecHitSrc = "hltSiPhase2RecHits",
    useOTExtension = True
)
