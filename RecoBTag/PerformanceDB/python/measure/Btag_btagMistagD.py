import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_MISTAGCSVLD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVLD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVLD_T'),
    WorkingPointName = cms.string('MISTAGCSVLD_WP')
)

BtagPerformanceESProducer_MISTAGCSVMD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVMD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVMD_T'),
    WorkingPointName = cms.string('MISTAGCSVMD_WP')
)

BtagPerformanceESProducer_MISTAGCSVTD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGCSVTD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGCSVTD_T'),
    WorkingPointName = cms.string('MISTAGCSVTD_WP')
)

BtagPerformanceESProducer_MISTAGJPLD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPLD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPLD_T'),
    WorkingPointName = cms.string('MISTAGJPLD_WP')
)

BtagPerformanceESProducer_MISTAGJPMD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPMD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPMD_T'),
    WorkingPointName = cms.string('MISTAGJPMD_WP')
)

BtagPerformanceESProducer_MISTAGJPTD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGJPTD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGJPTD_T'),
    WorkingPointName = cms.string('MISTAGJPTD_WP')
)

BtagPerformanceESProducer_MISTAGTCHPTD = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPTD'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MISTAGTCHPTD_T'),
    WorkingPointName = cms.string('MISTAGTCHPTD_WP')
)
