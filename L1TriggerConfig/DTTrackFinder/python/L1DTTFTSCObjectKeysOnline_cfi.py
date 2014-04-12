import FWCore.ParameterSet.Config as cms

L1DTTFTSCObjectKeysOnline = cms.ESProducer("DTTFTSCObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('DTTF'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


