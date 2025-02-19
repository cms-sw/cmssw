import FWCore.ParameterSet.Config as cms

hiSelectedProtoTracks = cms.EDFilter("HIProtoTrackSelection",
    src = cms.InputTag("hiPixel3ProtoTracks"),
	VertexCollection = cms.InputTag("hiPixelMedianVertex"),
        beamSpotLabel = cms.InputTag("offlineBeamSpot"),
	ptMin = cms.double(0.0),
	nSigmaZ = cms.double(5.0),
	minZCut = cms.double(0.2),
	maxD0Significance = cms.double(5.0)
)
