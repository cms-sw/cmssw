import FWCore.ParameterSet.Config as cms

L1RPCBxOrConfigOnline = cms.ESProducer("L1RPCBxOrConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


# foo bar baz
# PdGnagCpCS9Em
# kDIK0ixgF9BVq
