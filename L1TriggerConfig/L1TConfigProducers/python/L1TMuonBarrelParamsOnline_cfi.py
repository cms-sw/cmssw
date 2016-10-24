import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonBarrel.fakeBmtfParams_cff import *

L1TMuonBarrelParamsOnlineProd = cms.ESProducer("L1TMuonBarrelParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
