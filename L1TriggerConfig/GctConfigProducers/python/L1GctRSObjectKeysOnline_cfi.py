import FWCore.ParameterSet.Config as cms

L1GctRSObjectKeysOnline = cms.ESProducer("L1GctRSObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string(''),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    enableL1GctChannelMask = cms.bool( True )
)
