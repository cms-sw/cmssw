import FWCore.ParameterSet.Config as cms

pixelTracksHitDoublets = cms.EDProducer("HitPairEDProducer",
    clusterCheck = cms.InputTag(""),
    layerPairs = cms.vuint32(0, 1, 2),
    maxElement = cms.uint32(50000000),
    maxElementTotal = cms.uint32(50000000),
    produceIntermediateHitDoublets = cms.bool(True),
    produceSeedingHitSets = cms.bool(False),
    seedingLayers = cms.InputTag("pixelTracksSeedLayers"),
    trackingRegions = cms.InputTag("pixelTracksTrackingRegions"),
    trackingRegionsSeedingLayers = cms.InputTag("")
)
