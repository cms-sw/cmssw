import FWCore.ParameterSet.Config as cms

L1RPCConeDefinitionOnline = cms.ESProducer("L1RPCConeDefinitionOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)


