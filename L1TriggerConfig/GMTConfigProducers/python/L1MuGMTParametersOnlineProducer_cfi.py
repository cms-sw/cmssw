import FWCore.ParameterSet.Config as cms

L1MuGMTParametersOnlineProducer = cms.ESProducer("L1MuGMTParametersOnlineProducer",
                                       onlineDB = cms.string("oracle://CMS_OMDS_LB/CMS_TRG_R"),
                                       onlineAuthentication = cms.string("."),
                                       forceGeneration = cms.bool(False),
                                       ignoreVersionMismatch = cms.bool(False)
)
