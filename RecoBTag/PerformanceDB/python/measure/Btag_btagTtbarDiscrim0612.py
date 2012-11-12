import FWCore.ParameterSet.Config as cms
BtagPerformanceESProducer_TTBARDISCRIMBTAGCSV = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGCSV'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARDISCRIMBTAGCSVtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARDISCRIMBTAGCSVwp_v8_offline')
)
BtagPerformanceESProducer_TTBARDISCRIMBTAGJP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGJP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARDISCRIMBTAGJPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARDISCRIMBTAGJPwp_v8_offline')
)
BtagPerformanceESProducer_TTBARDISCRIMBTAGJBP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGJBP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARDISCRIMBTAGJBPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARDISCRIMBTAGJBPwp_v8_offline')
)
BtagPerformanceESProducer_TTBARDISCRIMBTAGSSVHE = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGSSVHE'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARDISCRIMBTAGSSVHEtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARDISCRIMBTAGSSVHEwp_v8_offline')
)
BtagPerformanceESProducer_TTBARDISCRIMBTAGSSVHP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGSSVHP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARDISCRIMBTAGSSVHPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARDISCRIMBTAGSSVHPwp_v8_offline')
)
BtagPerformanceESProducer_TTBARDISCRIMBTAGTCHE = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGTCHE'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARDISCRIMBTAGTCHEtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARDISCRIMBTAGTCHEwp_v8_offline')
)
BtagPerformanceESProducer_TTBARDISCRIMBTAGTCHP = cms.ESProducer("BtagPerformanceESProducer",
# this is what it makes available
    ComponentName = cms.string('TTBARDISCRIMBTAGTCHP'),
# this is where it gets the payload from                                                
    PayloadName = cms.string('BTagTTBARDISCRIMBTAGTCHPtable_v8_offline'),
    WorkingPointName = cms.string('BTagTTBARDISCRIMBTAGTCHPwp_v8_offline')
)
