import FWCore.ParameterSet.Config as cms

ghostTrackVertexRecoBlock = cms.PSet(
	vertexReco = cms.PSet(
		finder = cms.string("gtvr"),
		maxFitChi2 = cms.double(10.0),
		mergeThreshold = cms.double(0.8),
		primcut = cms.double(1.8),
		seccut = cms.double(4.0),
		fitType = cms.string("SingleTracksWithGhostTrack")
	)
)
