import FWCore.ParameterSet.Config as cms

hltPhase2L3MuonPixelTracksHitDoublets = cms.EDProducer("HitPairEDProducer",
    clusterCheck = cms.InputTag("trackerClusterCheck"),
    layerPairs = cms.vuint32(0, 1, 2),
    maxElement = cms.uint32(5000000),
    maxElementTotal = cms.uint32(50000000),
    mightGet = cms.optional.untracked.vstring,
    produceIntermediateHitDoublets = cms.bool(True),
    produceSeedingHitSets = cms.bool(False),
    seedingLayers = cms.InputTag("hltPhase2L3MuonPixelTracksSeedLayers"),
    trackingRegions = cms.InputTag("hltPhase2L3MuonPixelTracksTrackingRegions"),
    trackingRegionsSeedingLayers = cms.InputTag("")
)
