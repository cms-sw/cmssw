import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonEndCap.fakeEmtfParams_cff import *

L1TMuonEndCapForestOnlineProd = cms.ESProducer("L1TMuonEndCapForestOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(False),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # any value has no effect on this particular producer
)
