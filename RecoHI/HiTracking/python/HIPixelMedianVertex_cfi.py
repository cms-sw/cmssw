import FWCore.ParameterSet.Config as cms

hiPixelMedianVertex = cms.EDFilter("HIPixelMedianVtxProducer",
    TrackCollection = cms.string('hiPixel3ProtoTracks'),
    PtMin = cms.double(1.0)
)


