#
import FWCore.ParameterSet.Config as cms

l1GtTriggerMaskVetoTechTrigOnline = cms.ESProducer("L1GtTriggerMaskVetoTechTrigConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    #
    PartitionNumber = cms.int32(0)
)

