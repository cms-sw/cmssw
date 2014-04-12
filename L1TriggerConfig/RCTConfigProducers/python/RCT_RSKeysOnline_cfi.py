import FWCore.ParameterSet.Config as cms
RCT_RSKeysOnline = cms.ESProducer("L1RCT_RSKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('RCT_'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    enableL1RCTChannelMask = cms.bool( True )
)
