import FWCore.ParameterSet.Config as cms

hltIter2Phase2L3FromL1TkMuonPixelHitDoublets = cms.EDProducer("HitPairEDProducer",
    clusterCheck = cms.InputTag("hltIter2Phase2L3FromL1TkMuonPixelClusterCheck"),
    layerPairs = cms.vuint32(0, 1),
    maxElement = cms.uint32(0),
    produceIntermediateHitDoublets = cms.bool(True),
    produceSeedingHitSets = cms.bool(False),
    seedingLayers = cms.InputTag("hltIter2Phase2L3FromL1TkMuonPixelLayerTriplets"),
    trackingRegions = cms.InputTag("hltPhase2L3FromL1TkMuonPixelTracksTrackingRegions"),
    trackingRegionsSeedingLayers = cms.InputTag("")
)
