import FWCore.ParameterSet.Config as cms

L1RCTParametersOnline = cms.ESProducer("L1RCTParametersOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


# foo bar baz
# VcAIVpbYWOYrE
# iYOztv1sLc72w
