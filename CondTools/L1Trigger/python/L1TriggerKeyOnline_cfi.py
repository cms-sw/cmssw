import FWCore.ParameterSet.Config as cms

L1TriggerKeyOnline = cms.ESProducer("L1TriggerKeyOnlineProd",
    onlineAuthentication = cms.string('.'),
    tscKey = cms.string('dummy'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    recordsToInclude = cms.vstring()
)


