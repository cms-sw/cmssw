import FWCore.ParameterSet.Config as cms

trackVertexArbitrator = cms.EDProducer("TrackVertexArbitrator",
    beamSpot = cms.InputTag("offlineBeamSpot"),
    dLenFraction = cms.double(0.333),
    dRCut = cms.double(0.4),
    distCut = cms.double(0.04),
    fitterRatio = cms.double(0.25),
    fitterSigmacut = cms.double(3),
    fitterTini = cms.double(256),
    maxTimeSignificance = cms.double(3.5),
    primaryVertices = cms.InputTag("offlinePrimaryVertices"),
    secondaryVertices = cms.InputTag("vertexMerger"),
    sigCut = cms.double(5),
    trackMinLayers = cms.int32(4),
    trackMinPixels = cms.int32(1),
    trackMinPt = cms.double(0.9),
    tracks = cms.InputTag("generalTracks")
)
