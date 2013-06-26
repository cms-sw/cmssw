import FWCore.ParameterSet.Config as cms
BtagPerformanceESProducer_TTBARMCBTAGCSV = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARMCBTAGCSV'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARMCBTAGCSVtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARMCBTAGCSVwp_v8_offline')
)
BtagPerformanceESProducer_TTBARMCBTAGJP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARMCBTAGJP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARMCBTAGJPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARMCBTAGJPwp_v8_offline')
)
BtagPerformanceESProducer_TTBARMCBTAGJBP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARMCBTAGJBP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARMCBTAGJBPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARMCBTAGJBPwp_v8_offline')
)
BtagPerformanceESProducer_TTBARMCBTAGSSVHE = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARMCBTAGSSVHE'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARMCBTAGSSVHEtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARMCBTAGSSVHEwp_v8_offline')
)
BtagPerformanceESProducer_TTBARMCBTAGSSVHP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARMCBTAGSSVHP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARMCBTAGSSVHPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARMCBTAGSSVHPwp_v8_offline')
)
BtagPerformanceESProducer_TTBARMCBTAGTCHE = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARMCBTAGTCHE'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARMCBTAGTCHEtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARMCBTAGTCHEwp_v8_offline')
)
BtagPerformanceESProducer_TTBARMCBTAGTCHP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARMCBTAGTCHP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARMCBTAGTCHPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARMCBTAGTCHPwp_v8_offline')
)
