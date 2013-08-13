import FWCore.ParameterSet.Config as cms


jvcParameters = cms.PSet(
	primaryVertices = cms.InputTag("offlinePrimaryVertices"),
	cut = cms.double(3.0),
	temperature = cms.double(1.5)
)
