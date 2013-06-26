import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_TTBARDISCRIMBTAGCSV = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGCSV'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARDISCRIMBTAGCSV_T'),
    WorkingPointName = cms.string('TTBARDISCRIMBTAGCSV_WP')
)

BtagPerformanceESProducer_TTBARDISCRIMBTAGJP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGJP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARDISCRIMBTAGJP_T'),
    WorkingPointName = cms.string('TTBARDISCRIMBTAGJP_WP')
)

BtagPerformanceESProducer_TTBARDISCRIMBTAGTCHP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGTCHP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TTBARDISCRIMBTAGTCHP_T'),
    WorkingPointName = cms.string('TTBARDISCRIMBTAGTCHP_WP')
)
