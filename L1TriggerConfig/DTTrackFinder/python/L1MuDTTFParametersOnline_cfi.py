import FWCore.ParameterSet.Config as cms

L1MuDTTFParametersOnline = cms.ESProducer("DTTFParametersOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


# foo bar baz
# TXayvMybrgA2q
