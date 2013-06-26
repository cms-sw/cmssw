import FWCore.ParameterSet.Config as cms
BtagPerformanceESProducer_BTAGCSVL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGCSVL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGCSVLtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGCSVLwp_v6_offline')
)
BtagPerformanceESProducer_BTAGCSVM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGCSVM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGCSVMtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGCSVMwp_v6_offline')
)
BtagPerformanceESProducer_BTAGCSVT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGCSVT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGCSVTtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGCSVTwp_v6_offline')
)
BtagPerformanceESProducer_BTAGJBPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGJBPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGJBPLtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGJBPLwp_v6_offline')
)
BtagPerformanceESProducer_BTAGJBPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGJBPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGJBPMtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGJBPMwp_v6_offline')
)
BtagPerformanceESProducer_BTAGJBPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGJBPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGJBPTtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGJBPTwp_v6_offline')
)
BtagPerformanceESProducer_BTAGJPL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGJPL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGJPLtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGJPLwp_v6_offline')
)
BtagPerformanceESProducer_BTAGJPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGJPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGJPMtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGJPMwp_v6_offline')
)
BtagPerformanceESProducer_BTAGJPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGJPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGJPTtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGJPTwp_v6_offline')
)
BtagPerformanceESProducer_BTAGSSVHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGSSVHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGSSVHEMtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGSSVHEMwp_v6_offline')
)
BtagPerformanceESProducer_BTAGSSVHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGSSVHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGSSVHPTtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGSSVHPTwp_v6_offline')
)
BtagPerformanceESProducer_BTAGTCHEL = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGTCHEL'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGTCHELtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGTCHELwp_v6_offline')
)
BtagPerformanceESProducer_BTAGTCHEM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGTCHEM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGTCHEMtable_v7_offline'),
    WorkingPointName = cms.string('BTagBTAGTCHEMwp_v7_offline')
)
BtagPerformanceESProducer_BTAGTCHPM = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGTCHPM'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGTCHPMtable_v7_offline'),
    WorkingPointName = cms.string('BTagBTAGTCHPMwp_v7_offline')
)
BtagPerformanceESProducer_BTAGTCHPT = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('BTAGTCHPT'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagBTAGTCHPTtable_v6_offline'),
    WorkingPointName = cms.string('BTagBTAGTCHPTwp_v6_offline')
)
