import FWCore.ParameterSet.Config as cms

L1RPCObjectKeysOnline = cms.ESProducer("RPCObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('RPC'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    enableL1RPCConfig = cms.bool(True),
    enableL1RPCConeDefinition = cms.bool(True),
    enableL1RPCHsbConfig = cms.bool(True),
    enableL1RPCBxOrConfig = cms.bool(True)
)


