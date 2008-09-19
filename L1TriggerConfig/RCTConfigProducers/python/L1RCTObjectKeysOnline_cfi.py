import FWCore.ParameterSet.Config as cms

L1RCTObjectKeysOnline = cms.ESProducer("RCTObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('RCT'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


