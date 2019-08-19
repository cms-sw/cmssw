import FWCore.ParameterSet.Config as cms

TopLHEFilter = cms.EDFilter('TopLHEFilter',
jetPtMin = cms.double(350.),
jetR = cms.double(0.8),
src = cms.InputTag('externalLHEProducer')
)

