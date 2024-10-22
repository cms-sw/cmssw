import FWCore.ParameterSet.Config as cms

L1TMuonEndCapObjectKeysOnline = cms.ESProducer("L1TMuonEndCapObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel       = cms.string('EMTF'),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # nothrow guarantee if set to False: carry on no matter what
)
