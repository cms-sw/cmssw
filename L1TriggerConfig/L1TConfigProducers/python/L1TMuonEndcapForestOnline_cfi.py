import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff import *

L1TMuonEndcapForestOnlineProd = cms.ESProducer("L1TMuonEndcapForestOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
