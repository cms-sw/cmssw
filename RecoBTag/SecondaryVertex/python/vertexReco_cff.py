import FWCore.ParameterSet.Config as cms

vertexRecoBlock = cms.PSet(
	vertexReco = cms.PSet(
		seccut = cms.double(6.0),
		primcut = cms.double(1.8),
		smoothing = cms.bool(False),
		finder = cms.string('avr'),
		minweight = cms.double(0.5),
		weightthreshold = cms.double(0.001)
	)
)
