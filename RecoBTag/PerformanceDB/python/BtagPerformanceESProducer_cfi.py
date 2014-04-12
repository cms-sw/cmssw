import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_test = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('JetProbability_loose'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('JetProbability_loose'),
    WorkingPointName = cms.string('JetProbability_loose')
)


