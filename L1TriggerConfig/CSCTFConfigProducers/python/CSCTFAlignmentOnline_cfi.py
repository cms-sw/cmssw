import FWCore.ParameterSet.Config as cms
CSCTFAlignmentOnline = cms.ESProducer("CSCTFAlignmentOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(True),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)
