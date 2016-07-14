def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)
def customizeMinPtForHitRecoveryInGluedDet(process,value):
   for esp in esproducers_by_type(process, "Chi2MeasurementEstimatorESProducer", "Chi2ChargeMeasurementEstimatorESProducer"):
       esp.MinPtForHitRecoveryInGluedDet = 100000
def customizeHitRecoveryInGluedDetOff(process):
   for esp in esproducers_by_type(process, "Chi2MeasurementEstimatorESProducer", "Chi2ChargeMeasurementEstimatorESProducer"):
       customizeMinPtForHitRecoveryInGluedDet(process,1000000)

   return process

