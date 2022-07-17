import FWCore.ParameterSet.Config as cms

from Calibration.HcalCalibAlgos.hcalTestThreshold_cfi import *

from Configuration.Eras.Modifier_run2_ECAL_2017_cff import run2_ECAL_2017

run2_ECAL_2017.toModify(hcalTestThreshold,
  EBHitEnergyThreshold    = 0.18,
  EEHitEnergyThreshold0   = -206.074,
  EEHitEnergyThreshold1   = 357.671,
  EEHitEnergyThreshold2   = -204.978,
  EEHitEnergyThreshold3   = 39.033,
  EEHitEnergyThresholdLow = 1.25,
  EEHitEnergyThresholdHigh= 10.0,
)

from Configuration.Eras.Modifier_run2_ECAL_2018_cff import run2_ECAL_2018

run2_ECAL_2018.toModify(hcalTestThreshold,
  EBHitEnergyThreshold    = 0.10,
  EEHitEnergyThreshold0   = -41.0664,
  EEHitEnergyThreshold1   = 68.795,
  EEHitEnergyThreshold2   = -38.1483,
  EEHitEnergyThreshold3   = 7.04303,
  EEHitEnergyThresholdLow = 0.11,
  EEHitEnergyThresholdHigh= 15.4,
)
