import FWCore.ParameterSet.Config as cms

pixel3Vertices = cms.EDFilter("PixelVertexProducerMedian",
    TrackCollection = cms.string('pixelTracks'),
    PtMin = cms.double(1.0)
)


