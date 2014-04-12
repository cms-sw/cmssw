import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_MISTAGCSVLABCD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVLABCD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVLABCD_T'),
    WorkingPointName = cms.string('MISTAGCSVLABCD_WP')
)

BtagPerformanceESProducer_MISTAGCSVMABCD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVMABCD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVMABCD_T'),
    WorkingPointName = cms.string('MISTAGCSVMABCD_WP')
)

BtagPerformanceESProducer_MISTAGCSVTABCD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVTABCD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVTABCD_T'),
    WorkingPointName = cms.string('MISTAGCSVTABCD_WP')
)

BtagPerformanceESProducer_MISTAGJPLABCD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPLABCD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPLABCD_T'),
    WorkingPointName = cms.string('MISTAGJPLABCD_WP')
)

BtagPerformanceESProducer_MISTAGJPMABCD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPMABCD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPMABCD_T'),
    WorkingPointName = cms.string('MISTAGJPMABCD_WP')
)

BtagPerformanceESProducer_MISTAGJPTABCD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPTABCD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPTABCD_T'),
    WorkingPointName = cms.string('MISTAGJPTABCD_WP')
)

BtagPerformanceESProducer_MISTAGTCHPTABCD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPTABCD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGTCHPTABCD_T'),
    WorkingPointName = cms.string('MISTAGTCHPTABCD_WP')
)
