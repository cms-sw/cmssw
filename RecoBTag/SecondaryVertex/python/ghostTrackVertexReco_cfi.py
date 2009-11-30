import FWCore.ParameterSet.Config as cms

ghostTrackVertexRecoBlock = cms.PSet(
	vertexReco = cms.PSet(
		finder = cms.string("gtvr"),
		maxFitChi2 = cms.double(10.0),
		mergeThreshold = cms.double(2.7),
		primcut = cms.double(1.8),
		seccut = cms.double(5.0),
		fitType = cms.string("SingleTracksWithGhostTrack")
	)
)
