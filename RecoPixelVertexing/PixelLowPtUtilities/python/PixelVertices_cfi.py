import FWCore.ParameterSet.Config as cms

pixelVertices = cms.EDFilter("PixelVertexProducerMedian",
    TrackCollection = cms.string('pixelTracks'),
    PtMin = cms.double(1.0)
)


