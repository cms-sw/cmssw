import FWCore.ParameterSet.Config as cms
CSCTFObjectKeysOnline = cms.ESProducer("CSCTFObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('CSCTF'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    enableConfiguration = cms.bool( True ),
    enablePtLut = cms.bool( True )
)
