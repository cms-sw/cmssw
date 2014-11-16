import FWCore.ParameterSet.Config as cms

simpleSecondaryVertex2TrkComputer = cms.ESProducer("SimpleSecondaryVertexESProducer",
	use3d = cms.bool(True),
	unBoost = cms.bool(False),
	useSignificance = cms.bool(True),
	minTracks = cms.uint32(2)
)
