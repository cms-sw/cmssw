import FWCore.ParameterSet.Config as cms


pvMonitor = cms.EDProducer("PrimaryVertexMonitor",
                       vertexLabel = cms.InputTag("offlinePrimaryVertices"),
                       beamSpotLabel = cms.InputTag("offlineBeamSpot")
)
