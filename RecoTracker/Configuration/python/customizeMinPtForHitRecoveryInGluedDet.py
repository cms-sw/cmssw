def esproducers_by_type(process, *types):
    return (module for module in process._Process__esproducers.values() if module._TypedParameterizable__type in types)
def customizeMinPtForHitRecoveryInGluedDet(process):
   for esp in esproducers_by_type(process, "Chi2MeasurementEstimatorESProducer", "Chi2ChargeMeasurementEstimatorESProducer"):
       esp.MinPtForHitRecoveryInGluedDet = 100000
   return process

