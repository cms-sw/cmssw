#
import FWCore.ParameterSet.Config as cms

l1GtPrescaleFactorsTechTrigOnline = cms.ESProducer("L1GtPrescaleFactorsTechTrigConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R')
)

