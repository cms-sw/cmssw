import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_MISTAGCSVLAB = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVLAB'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVLAB_T'),
    WorkingPointName = cms.string('MISTAGCSVLAB_WP')
)

BtagPerformanceESProducer_MISTAGCSVMAB = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVMAB'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVMAB_T'),
    WorkingPointName = cms.string('MISTAGCSVMAB_WP')
)

BtagPerformanceESProducer_MISTAGCSVTAB = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVTAB'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVTAB_T'),
    WorkingPointName = cms.string('MISTAGCSVTAB_WP')
)

BtagPerformanceESProducer_MISTAGJPLAB = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPLAB'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPLAB_T'),
    WorkingPointName = cms.string('MISTAGJPLAB_WP')
)

BtagPerformanceESProducer_MISTAGJPMAB = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPMAB'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPMAB_T'),
    WorkingPointName = cms.string('MISTAGJPMAB_WP')
)

BtagPerformanceESProducer_MISTAGJPTAB = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPTAB'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPTAB_T'),
    WorkingPointName = cms.string('MISTAGJPTAB_WP')
)

BtagPerformanceESProducer_MISTAGTCHPTAB = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPTAB'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGTCHPTAB_T'),
    WorkingPointName = cms.string('MISTAGTCHPTAB_WP')
)
