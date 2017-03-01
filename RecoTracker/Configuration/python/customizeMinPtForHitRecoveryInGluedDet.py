import FWCore.ParameterSet.Config as cms
def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)
def customizeMinPtForHitRecoveryInGluedDet(process,value):
   for esp in esproducers_by_type(process, "Chi2MeasurementEstimatorESProducer", "Chi2ChargeMeasurementEstimatorESProducer"):
       esp.MinPtForHitRecoveryInGluedDet = cms.double(value)
   return process
def customizeHitRecoveryInGluedDetOff(process):
   return customizeMinPtForHitRecoveryInGluedDet(process,1000000)


