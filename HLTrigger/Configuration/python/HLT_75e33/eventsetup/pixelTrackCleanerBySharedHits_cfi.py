import FWCore.ParameterSet.Config as cms

pixelTrackCleanerBySharedHits = cms.ESProducer("PixelTrackCleanerBySharedHitsESProducer",
    ComponentName = cms.string('pixelTrackCleanerBySharedHits'),
    appendToDataLabel = cms.string(''),
    useQuadrupletAlgo = cms.bool(False)
)
