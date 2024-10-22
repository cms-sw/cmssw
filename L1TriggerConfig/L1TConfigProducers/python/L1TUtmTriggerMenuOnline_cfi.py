import FWCore.ParameterSet.Config as cms

L1TUtmTriggerMenuOnlineProd = cms.ESProducer("L1TUtmTriggerMenuOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(True),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
    # menu producer must be transaction safe otherwise everyone is screwed
)
