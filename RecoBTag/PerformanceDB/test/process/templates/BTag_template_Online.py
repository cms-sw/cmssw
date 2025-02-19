import FWCore.ParameterSet.Config as cms
BtagPerformanceESProducer_TEMPLATE = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TEMPLATE'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTEMPLATEtable_VERSION_offline'),
    WorkingPointName = cms.string('BTagTEMPLATEwp_VERSION_offline')
)
