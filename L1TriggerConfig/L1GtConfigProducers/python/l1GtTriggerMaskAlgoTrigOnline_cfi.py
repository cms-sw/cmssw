#
import FWCore.ParameterSet.Config as cms

l1GtTriggerMaskAlgoTrigOnline = cms.ESProducer("L1GtTriggerMaskAlgoTrigConfigOnlineProd",
    onlineAuthentication = cms.string('.'),
    forceGeneration = cms.bool(False),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    #
    PartitionNumber = cms.int32(0)

)

# foo bar baz
