import FWCore.ParameterSet.Config as cms
import L1Trigger.RegionalCaloTrigger.rctCalibrationCommon_cff as rctCalib

rctGenCalibrator = cms.EDProducer("L1RCTGenCalibrator",
                                  rctCalib.common
                                  )
