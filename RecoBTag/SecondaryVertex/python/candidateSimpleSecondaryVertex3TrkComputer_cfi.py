import FWCore.ParameterSet.Config as cms

candidateSimpleSecondaryVertex3TrkComputer = cms.ESProducer("CandidateSimpleSecondaryVertexESProducer",
	use3d = cms.bool(True),
	unBoost = cms.bool(False),
	useSignificance = cms.bool(True),
	minTracks = cms.uint32(3)
)
