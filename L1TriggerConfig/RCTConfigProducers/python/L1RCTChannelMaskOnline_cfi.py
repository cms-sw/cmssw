import FWCore.ParameterSet.Config as cms

L1RCTChannelMaskOnline = cms.ESProducer("L1RCTChannelMaskOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


# foo bar baz
