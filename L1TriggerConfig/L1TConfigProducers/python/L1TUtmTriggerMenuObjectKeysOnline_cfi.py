import FWCore.ParameterSet.Config as cms

L1TUtmTriggerMenuObjectKeysOnline = cms.ESProducer("L1TUtmTriggerMenuObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel       = cms.string('uGT'),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
    # menu producer must be transaction safe otherwise everyone is screwed
)

