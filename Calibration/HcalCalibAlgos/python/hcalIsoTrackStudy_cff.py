import FWCore.ParameterSet.Config as cms

from Calibration.HcalCalibAlgos.hcalIsoTrackStudy_cfi import *

from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017

run2_ECAL_2017.toModify(hcalIsoTrackStudy,
  EBHitEnergyThreshold    = cms.double(0.18),
  EEHitEnergyThreshold0   = cms.double(-206.074),
  EEHitEnergyThreshold1   = cms.double(357.671),
  EEHitEnergyThreshold2   = cms.double(-204.978),
  EEHitEnergyThreshold3   = cms.double(39.033),
  EEHitEnergyThresholdLow = cms.double(1.25),
  EEHitEnergyThresholdHigh= cms.double(10.0),
)

from Configuration.Eras.Modifier_run2_HCAL_2018_cff import run2_HCAL_2018

run2_HCAL_2018.toModify(hcalIsoTrackStudy,
  EBHitEnergyThreshold    = cms.double(0.10),
  EEHitEnergyThreshold0   = cms.double(-41.0664),
  EEHitEnergyThreshold1   = cms.double(68.795),
  EEHitEnergyThreshold2   = cms.double(-38.1483),
  EEHitEnergyThreshold3   = cms.double(7.04303),
  EEHitEnergyThresholdLow = cms.double(0.11),
  EEHitEnergyThresholdHigh= cms.double(15.4),
)
