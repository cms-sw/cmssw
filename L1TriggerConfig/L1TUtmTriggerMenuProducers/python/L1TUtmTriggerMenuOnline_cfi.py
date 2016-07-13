import FWCore.ParameterSet.Config as cms

from L1TriggerConfig.L1TUtmTriggerMenuProducers.L1TUtmTriggerMenuObjectKeysOnline_cfi import *

L1TUtmTriggerMenuOnlineProd = cms.ESProducer("L1TUtmTriggerMenuOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)

#es_prefer = cms.ESPrefer("L1TCaloParamsStage1OnlineProd","")
