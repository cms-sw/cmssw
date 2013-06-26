import FWCore.ParameterSet.Config as cms


pvMonitor = cms.EDAnalyzer("PrimaryVertexMonitor",
                       vertexLabel = cms.InputTag("offlinePrimaryVertices"),
                       beamSpotLabel = cms.InputTag("offlineBeamSpot")
)
