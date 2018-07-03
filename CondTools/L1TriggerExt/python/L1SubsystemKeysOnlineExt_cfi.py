import FWCore.ParameterSet.Config as cms

L1SubsystemKeysOnlineExt = cms.ESProducer("L1SubsystemKeysOnlineProdExt",
    onlineAuthentication = cms.string('.'),
    tscKey = cms.string('dummy'),
    rsKey  = cms.string('dummy'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    forceGeneration = cms.bool(False)
)


