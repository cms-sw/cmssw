import FWCore.ParameterSet.Config as cms
BtagPerformanceESProducer_NAME = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('NAME'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('NAME_T'),
    WorkingPointName = cms.string('NAME_WP')
)
