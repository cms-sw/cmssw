import FWCore.ParameterSet.Config as cms

L1GctRSKeysOnline = cms.ESProducer("L1GctRSKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string(''),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    enableMYOBJECT = cms.bool( True )
)
