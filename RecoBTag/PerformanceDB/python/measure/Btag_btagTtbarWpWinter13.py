import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_TTBARWPBTAGCSVL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARWPBTAGCSVL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARWPBTAGCSVL_T'),
    WorkingPointName = cms.string('TTBARWPBTAGCSVL_WP')
)

BtagPerformanceESProducer_TTBARWPBTAGCSVM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARWPBTAGCSVM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARWPBTAGCSVM_T'),
    WorkingPointName = cms.string('TTBARWPBTAGCSVM_WP')
)

BtagPerformanceESProducer_TTBARWPBTAGCSVT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARWPBTAGCSVT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARWPBTAGCSVT_T'),
    WorkingPointName = cms.string('TTBARWPBTAGCSVT_WP')
)

BtagPerformanceESProducer_TTBARWPBTAGJPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARWPBTAGJPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARWPBTAGJPL_T'),
    WorkingPointName = cms.string('TTBARWPBTAGJPL_WP')
)

BtagPerformanceESProducer_TTBARWPBTAGJPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARWPBTAGJPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARWPBTAGJPM_T'),
    WorkingPointName = cms.string('TTBARWPBTAGJPM_WP')
)

BtagPerformanceESProducer_TTBARWPBTAGJPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARWPBTAGJPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARWPBTAGJPT_T'),
    WorkingPointName = cms.string('TTBARWPBTAGJPT_WP')
)

BtagPerformanceESProducer_TTBARWPBTAGTCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARWPBTAGTCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARWPBTAGTCHPT_T'),
    WorkingPointName = cms.string('TTBARWPBTAGTCHPT_WP')
)
