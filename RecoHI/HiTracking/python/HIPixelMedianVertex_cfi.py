import FWCore.ParameterSet.Config as cms

hiPixelMedianVertex = cms.EDFilter("HIPixelMedianVtxProducer",
    TrackCollection = cms.string('hiPixel3ProtoTracks'),
    PtMin = cms.double(0.0) # selection already made in pixel track filter
)


