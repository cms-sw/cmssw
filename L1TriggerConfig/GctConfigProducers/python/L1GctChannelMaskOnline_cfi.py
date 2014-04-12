import FWCore.ParameterSet.Config as cms

L1GctChannelMaskOnline = cms.ESProducer("L1GctChannelMaskOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
