import FWCore.ParameterSet.Config as cms

L1TMuonEndcapObjectKeysOnline = cms.ESProducer("L1TMuonEndcapObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('EMTF'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
