#
import FWCore.ParameterSet.Config as cms

l1GtParametersOnline = cms.ESProducer("L1GtParametersConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)

