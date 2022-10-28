import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TMuonOverlap.fakeOmtfFwVersion_cff import *

L1TMuonOverlapFwVersionOnlineProd = cms.ESProducer("L1TMuonOverlapFwVersionOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(True),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)
