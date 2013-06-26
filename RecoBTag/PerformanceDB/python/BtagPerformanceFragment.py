import FWCore.ParameterSet.Config as cms

BtagPerformanceESProducer_1 = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('JetProbability_loose'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('JPL_T'),
    WorkingPointName = cms.string('JPL_WP')
)


BtagPerformanceESProducer_2 = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('JetProbability_medium'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('JPM_T'),
    WorkingPointName = cms.string('JPM_WP')
)


BtagPerformanceESProducer_3 = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('JetProbability_tight'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('JPT_T'),
    WorkingPointName = cms.string('JPT_WP')
)


BtagPerformanceESProducer_4 = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TrackCountingHighEfficiency_loose'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TCHEL_T'),
    WorkingPointName = cms.string('TCHEL_WP')
)

BtagPerformanceESProducer_5 = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TrackCountingHighEfficiency_medium'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TCHEM_T'),
    WorkingPointName = cms.string('TCHEM_WP')
)

BtagPerformanceESProducer_5 = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TrackCountingHighPurity_tight'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('TCHPT_T'),
    WorkingPointName = cms.string('TCHPT_WP')
)





    





