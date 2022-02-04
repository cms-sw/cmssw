import FWCore.ParameterSet.Config as cms

hltElePixelHitDoubletsL1Seeded = cms.EDProducer("HitPairEDProducer",
    clusterCheck = cms.InputTag(""),
    layerPairs = cms.vuint32(0),
    maxElement = cms.uint32(0),
    maxElementTotal = cms.uint32(50000000),
    produceIntermediateHitDoublets = cms.bool(True),
    produceSeedingHitSets = cms.bool(True),
    seedingLayers = cms.InputTag("hltPixelLayerPairsL1Seeded"),
    trackingRegions = cms.InputTag("hltEleSeedsTrackingRegionsL1Seeded"),
    trackingRegionsSeedingLayers = cms.InputTag("")
)
