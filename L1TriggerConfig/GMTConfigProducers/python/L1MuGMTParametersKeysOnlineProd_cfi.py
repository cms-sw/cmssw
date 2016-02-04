import FWCore.ParameterSet.Config as cms

L1MuGMTParametersKeysOnlineProd = cms.ESProducer("L1MuGMTParametersKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('GMT'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')                                                
)


