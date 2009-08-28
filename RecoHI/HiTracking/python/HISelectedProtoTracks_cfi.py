import FWCore.ParameterSet.Config as cms

hiSelectedProtoTracks = cms.EDFilter("HIProtoTrackSelection",
    src = cms.InputTag("hiPixel3ProtoTracks"),
	VertexCollection = cms.string("hiPixelMedianVertex"),
	ptMin = cms.double(1.0),
	nSigmaZ = cms.double(5.0),
	minZCut = cms.double(0.2),
	maxD0Significance = cms.double(5.0)
)
