import FWCore.ParameterSet.Config as cms
L1GctTSCObjectKeysOnline = cms.ESProducer("L1GctTSCObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('GCT'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
