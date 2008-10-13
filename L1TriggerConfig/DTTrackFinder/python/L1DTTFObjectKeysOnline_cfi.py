import FWCore.ParameterSet.Config as cms

L1DTTFObjectKeysOnline = cms.ESProducer("DTTFObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('DTTF'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


