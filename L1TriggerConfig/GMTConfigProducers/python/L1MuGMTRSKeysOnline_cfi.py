import FWCore.ParameterSet.Config as cms
L1MuGMTRSKeysOnline = cms.ESProducer("L1MuGMTRSKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('L1MuGMT'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    enableL1MuGMTChannelMask = cms.bool( True )
)
