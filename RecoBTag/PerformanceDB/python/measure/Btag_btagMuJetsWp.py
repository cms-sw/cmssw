import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_MUJETSWPBTAGCSVL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MUJETSWPBTAGCSVL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MUJETSWPBTAGCSVL_T'),
    WorkingPointName = cms.string('MUJETSWPBTAGCSVL_WP')
)

BtagPerformanceESProducer_MUJETSWPBTAGCSVM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MUJETSWPBTAGCSVM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MUJETSWPBTAGCSVM_T'),
    WorkingPointName = cms.string('MUJETSWPBTAGCSVM_WP')
)

BtagPerformanceESProducer_MUJETSWPBTAGCSVT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MUJETSWPBTAGCSVT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MUJETSWPBTAGCSVT_T'),
    WorkingPointName = cms.string('MUJETSWPBTAGCSVT_WP')
)

BtagPerformanceESProducer_MUJETSWPBTAGJPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MUJETSWPBTAGJPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MUJETSWPBTAGJPL_T'),
    WorkingPointName = cms.string('MUJETSWPBTAGJPL_WP')
)

BtagPerformanceESProducer_MUJETSWPBTAGJPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MUJETSWPBTAGJPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MUJETSWPBTAGJPM_T'),
    WorkingPointName = cms.string('MUJETSWPBTAGJPM_WP')
)

BtagPerformanceESProducer_MUJETSWPBTAGJPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MUJETSWPBTAGJPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MUJETSWPBTAGJPT_T'),
    WorkingPointName = cms.string('MUJETSWPBTAGJPT_WP')
)

BtagPerformanceESProducer_MUJETSWPBTAGTCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MUJETSWPBTAGTCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('MUJETSWPBTAGTCHPT_T'),
    WorkingPointName = cms.string('MUJETSWPBTAGTCHPT_WP')
)
