import FWCore.ParameterSet.Config as cms

# Legacy track converter for vector-hit-based tracking
# Converts SoA tracks to legacy reco::Track with vector hit expansion
hltPhase2PixelTracksWithVectorHits = cms.EDProducer("PixelTrackProducerFromSoAAlpaka",
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    trackSrc = cms.InputTag("hltPhase2PixelTrackTorchHighPuritySelector"),
    pixelRecHitLegacySrc = cms.InputTag("hltSiPixelRecHits"),
    outerTrackerRecHitSrc = cms.InputTag("hltSiPhase2RecHits"),
    outerTrackerRecHitSoAConverterSrc = cms.InputTag("phase2OTRecHitsSoAConverter"),   # not read with expandVectorHits
    otRecHitsSoASrc = cms.InputTag("hltCAOTRecHitsSoA"),
    vectorHitsSoASrc = cms.InputTag("hltPhase2OTVectorHitsSoA"),
    minNumberOfHits = cms.int32(0),
    minQuality = cms.string('tight'),
    useOTExtension = cms.bool(True),
    expandVectorHits = cms.bool(True),
    requireQuadsFromConsecutiveLayers = cms.bool(False)
)
