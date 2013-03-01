
import FWCore.ParameterSet.Config as cms

eeNoiseFilter = cms.EDFilter(
  "EENoiseFilter",
  EBRecHitSource = cms.InputTag('reducedEcalRecHitsEB'),
  EERecHitSource = cms.InputTag('reducedEcalRecHitsEE'),
  Slope     = cms.double(2),
  Intercept = cms.double(1000),
  taggingMode = cms.bool(False),
  debug = cms.bool(False),
)
