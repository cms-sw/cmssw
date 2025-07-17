import FWCore.ParameterSet.Config as cms

hltHighPtTripletStepHitDoublets = cms.EDProducer("HitPairEDProducer",
    clusterCheck = cms.InputTag("hltTrackerClusterCheck"),
    layerPairs = cms.vuint32(0, 1),
    maxElement = cms.uint32(50000000),
    maxElementTotal = cms.uint32(50000000),
    mightGet = cms.optional.untracked.vstring,
    produceIntermediateHitDoublets = cms.bool(True),
    produceSeedingHitSets = cms.bool(False),
    seedingLayers = cms.InputTag("hltHighPtTripletStepSeedLayers"),
    trackingRegions = cms.InputTag("hltPhase2PixelTracksAndHighPtStepTrackingRegions"),
    trackingRegionsSeedingLayers = cms.InputTag("")
)

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
phase2_hlt_vertexTrimming.toModify(hltHighPtTripletStepHitDoublets, trackingRegions = "hltTrackingRegionFromTrimmedVertices")
