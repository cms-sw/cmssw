
import FWCore.ParameterSet.Config as cms

ecalLaserCorrFilter = cms.EDFilter(
  "EcalLaserCorrFilter",
  EBRecHitSource = cms.InputTag("reducedEcalRecHitsEB"),
  EERecHitSource = cms.InputTag("reducedEcalRecHitsEE"),
  EBLaserMIN     = cms.double(0.3),
  EELaserMIN     = cms.double(0.3),
  EBLaserMAX     = cms.double(3.0),
  EELaserMAX     = cms.double(8.0),
  EBEnegyMIN     = cms.double(10.0),
  EEEnegyMIN     = cms.double(10.0),
  taggingMode    = cms.bool(False),
  Debug          = cms.bool(False)
)
