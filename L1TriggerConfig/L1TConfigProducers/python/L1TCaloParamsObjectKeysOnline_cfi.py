import FWCore.ParameterSet.Config as cms

L1TCaloParamsObjectKeysOnline = cms.ESProducer("L1TCaloParamsObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel       = cms.string('CALO'),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # any value has no effect on this particular producer
)

