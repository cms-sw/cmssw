import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_MISTAGCSVLC = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVLC'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVLC_T'),
    WorkingPointName = cms.string('MISTAGCSVLC_WP')
)

BtagPerformanceESProducer_MISTAGCSVMC = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVMC'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVMC_T'),
    WorkingPointName = cms.string('MISTAGCSVMC_WP')
)

BtagPerformanceESProducer_MISTAGCSVTC = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVTC'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVTC_T'),
    WorkingPointName = cms.string('MISTAGCSVTC_WP')
)

BtagPerformanceESProducer_MISTAGJPLC = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPLC'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPLC_T'),
    WorkingPointName = cms.string('MISTAGJPLC_WP')
)

BtagPerformanceESProducer_MISTAGJPMC = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPMC'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPMC_T'),
    WorkingPointName = cms.string('MISTAGJPMC_WP')
)

BtagPerformanceESProducer_MISTAGJPTC = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPTC'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPTC_T'),
    WorkingPointName = cms.string('MISTAGJPTC_WP')
)

BtagPerformanceESProducer_MISTAGTCHPTC = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPTC'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGTCHPTC_T'),
    WorkingPointName = cms.string('MISTAGTCHPTC_WP')
)
