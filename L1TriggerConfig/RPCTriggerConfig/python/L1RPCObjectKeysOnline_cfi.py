import FWCore.ParameterSet.Config as cms

L1RPCObjectKeysOnline = cms.ESProducer("RPCObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('RPC'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


