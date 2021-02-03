import FWCore.ParameterSet.Config as cms

siPixelClusterShapeCache = cms.EDProducer("SiPixelClusterShapeCacheProducer",
    mightGet = cms.optional.untracked.vstring,
    onDemand = cms.bool(False),
    src = cms.InputTag("siPixelClusters")
)
