import FWCore.ParameterSet.Config as cms
def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)
def customizeMinPtForHitRecoveryInGluedDet(process,value):
   for esp in esproducers_by_type(process, "Chi2MeasurementEstimatorESProducer", "Chi2ChargeMeasurementEstimatorESProducer"):
       esp.MinPtForHitRecoveryInGluedDet = cms.double(value)
   return process
def customizeHitRecoveryInGluedDetOff(process):
   return customizeMinPtForHitRecoveryInGluedDet(process,1000000)
def customizeHitRecoveryInGluedDetOn(process):
   process = customizeMinPtForHitRecoveryInGluedDet(process,0.9)
   if hasattr(process, "Chi2MeasurementEstimatorForP5"): # keep disabled for cosmics
       process.Chi2MeasurementEstimatorForP5.MinPtForHitRecoveryInGluedDet = 100000
   return process


