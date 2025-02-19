import FWCore.ParameterSet.Config as cms
BtagPerformanceESProducer_MISTAGSSVHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('MISTAGSSVHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagMISTAGSSVHEMtable_v2_offline'),
    WorkingPointName = cms.string('BTagMISTAGSSVHEMwp_v2_offline')
)
