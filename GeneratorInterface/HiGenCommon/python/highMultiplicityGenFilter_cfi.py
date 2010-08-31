import FWCore.ParameterSet.Config as cms

highMultiplicityGenFilter = cms.EDFilter("HighMultiplicityGenFilter",
  ptMin = cms.untracked.double(0.4),
  etaMax = cms.untracked.double(2.4),
  nMin = cms.untracked.int32(150)
)

