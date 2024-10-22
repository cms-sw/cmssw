import FWCore.ParameterSet.Config as cms

L1TMuonOverlapParamsOnlineProd = cms.ESProducer("L1TMuonOverlapParamsOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration      = cms.bool(False),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)
