import FWCore.ParameterSet.Config as cms

hltPhase2L3FromL1TkMuonPixelTracksHitDoublets = cms.EDProducer("HitPairEDProducer",
    clusterCheck = cms.InputTag(""),
    layerPairs = cms.vuint32(0, 1, 2),
    maxElement = cms.uint32(0),
    produceIntermediateHitDoublets = cms.bool(True),
    produceSeedingHitSets = cms.bool(False),
    seedingLayers = cms.InputTag("hltPhase2L3FromL1TkMuonPixelLayerQuadruplets"),
    trackingRegions = cms.InputTag("hltPhase2L3FromL1TkMuonPixelTracksTrackingRegions"),
    trackingRegionsSeedingLayers = cms.InputTag("")
)
