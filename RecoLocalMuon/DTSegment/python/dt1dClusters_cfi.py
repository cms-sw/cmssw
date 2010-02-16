import FWCore.ParameterSet.Config as cms

# module to build the 1d Clusters

dt1DClusters = cms.EDProducer("DTClusterer",
    debug = cms.untracked.bool(False),
    minLayers = cms.uint32(3),
    minHits = cms.uint32(3),
    recHits1DLabel = cms.InputTag("dt1DRecHits")
)

