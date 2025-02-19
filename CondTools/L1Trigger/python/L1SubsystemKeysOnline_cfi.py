import FWCore.ParameterSet.Config as cms

L1SubsystemKeysOnline = cms.ESProducer("L1SubsystemKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    tscKey = cms.string('dummy'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    forceGeneration = cms.bool(False)
)


