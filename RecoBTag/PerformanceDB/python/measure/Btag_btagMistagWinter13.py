import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_MISTAGCSVL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVL_T'),
    WorkingPointName = cms.string('MISTAGCSVL_WP')
)

BtagPerformanceESProducer_MISTAGCSVM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVM_T'),
    WorkingPointName = cms.string('MISTAGCSVM_WP')
)

BtagPerformanceESProducer_MISTAGCSVT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVT_T'),
    WorkingPointName = cms.string('MISTAGCSVT_WP')
)

BtagPerformanceESProducer_MISTAGCSVSLV1L = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVSLV1L'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVSLV1L_T'),
    WorkingPointName = cms.string('MISTAGCSVSLV1L_WP')
)

BtagPerformanceESProducer_MISTAGCSVSLV1M = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVSLV1M'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVSLV1M_T'),
    WorkingPointName = cms.string('MISTAGCSVSLV1M_WP')
)

BtagPerformanceESProducer_MISTAGCSVSLV1T = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVSLV1T'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVSLV1T_T'),
    WorkingPointName = cms.string('MISTAGCSVSLV1T_WP')
)

BtagPerformanceESProducer_MISTAGCSVV1L = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVV1L'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVV1L_T'),
    WorkingPointName = cms.string('MISTAGCSVV1L_WP')
)

BtagPerformanceESProducer_MISTAGCSVV1M = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVV1M'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVV1M_T'),
    WorkingPointName = cms.string('MISTAGCSVV1M_WP')
)

BtagPerformanceESProducer_MISTAGCSVV1T = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVV1T'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVV1T_T'),
    WorkingPointName = cms.string('MISTAGCSVV1T_WP')
)

BtagPerformanceESProducer_MISTAGJPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPL_T'),
    WorkingPointName = cms.string('MISTAGJPL_WP')
)

BtagPerformanceESProducer_MISTAGJPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPM_T'),
    WorkingPointName = cms.string('MISTAGJPM_WP')
)

BtagPerformanceESProducer_MISTAGJPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPT_T'),
    WorkingPointName = cms.string('MISTAGJPT_WP')
)

BtagPerformanceESProducer_MISTAGTCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGTCHPT_T'),
    WorkingPointName = cms.string('MISTAGTCHPT_WP')
)
