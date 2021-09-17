import FWCore.ParameterSet.Config as cms

LHEJetFilter = cms.EDFilter('LHEJetFilter',
jetPtMin = cms.double(350.),
jetR = cms.double(0.8),
src = cms.InputTag('externalLHEProducer')
)

