import FWCore.ParameterSet.Config as cms

L1TMuonOverlapObjectKeysOnline = cms.ESProducer("L1TMuonOverlapObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('OMTF'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
