import FWCore.ParameterSet.Config as cms
BtagPerformanceESProducer_MISTAGSSVHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGSSVHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGSSVHEMtable_v3_offline'),
    WorkingPointName = cms.string('BTagMISTAGSSVHEMwp_v3_offline')
)
BtagPerformanceESProducer_MISTAGSSVHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGSSVHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGSSVHPTtable_v3_offline'),
    WorkingPointName = cms.string('BTagMISTAGSSVHPTwp_v3_offline')
)
BtagPerformanceESProducer_MISTAGTCHEL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHEL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGTCHELtable_v3_offline'),
    WorkingPointName = cms.string('BTagMISTAGTCHELwp_v3_offline')
)
BtagPerformanceESProducer_MISTAGTCHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGTCHEMtable_v3_offline'),
    WorkingPointName = cms.string('BTagMISTAGTCHEMwp_v3_offline')
)
BtagPerformanceESProducer_MISTAGTCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGTCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGTCHPTtable_v3_offline'),
    WorkingPointName = cms.string('BTagMISTAGTCHPTwp_v3_offline')
)
