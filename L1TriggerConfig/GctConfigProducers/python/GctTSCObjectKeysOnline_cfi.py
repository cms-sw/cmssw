import FWCore.ParameterSet.Config as cms
GctTSCObjectKeysOnline = cms.ESProducer("Gct_TSCObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('Gct'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
