import FWCore.ParameterSet.Config as cms

L1TMuonGlobalObjectKeysOnline = cms.ESProducer("L1TMuonGlobalObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel       = cms.string('uGMT'),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # any value has no effect on this particular producer  
)

