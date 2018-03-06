import FWCore.ParameterSet.Config as cms

L1TGlobalPrescalesVetosObjectKeysOnline = cms.ESProducer("L1TGlobalPrescalesVetosObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel       = cms.string('uGTrs'),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # any value has no effect on this particular producer
)

