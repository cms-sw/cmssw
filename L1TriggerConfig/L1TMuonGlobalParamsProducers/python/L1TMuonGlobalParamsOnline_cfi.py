import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuon.fakeGmtParams_cff import *

L1TMuonGlobalParamsOnlineProd = cms.ESProducer("L1TMuonGlobalParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
