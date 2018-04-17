import FWCore.ParameterSet.Config as cms

L1TMuonBarrelObjectKeysOnline = cms.ESProducer("L1TMuonBarrelObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel       = cms.string('BMTF'),
    onlineDB             = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    transactionSafe      = cms.bool(True) # any value has no effect on this particular producer
)
