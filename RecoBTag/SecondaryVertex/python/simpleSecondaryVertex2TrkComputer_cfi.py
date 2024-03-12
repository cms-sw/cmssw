import FWCore.ParameterSet.Config as cms

simpleSecondaryVertex2TrkComputer = cms.ESProducer("SimpleSecondaryVertexESProducer",
	use3d = cms.bool(True),
	unBoost = cms.bool(False),
	useSignificance = cms.bool(True),
	minTracks = cms.uint32(2)
)
# foo bar baz
# UdfsxedlOB4XK
# GZymlTCIW0BSM
