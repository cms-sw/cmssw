# cfi file to produce the L1 GT keys for records managed via RUN SETTINGS

import FWCore.ParameterSet.Config as cms
l1GtRsObjectKeysOnline = cms.ESProducer("L1GtRsObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string(''),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    #
    PartitionNumber = cms.int32(0),
    EnableL1GtPrescaleFactorsAlgoTrig = cms.bool( True ),
    EnableL1GtPrescaleFactorsTechTrig = cms.bool( True ),
    EnableL1GtTriggerMaskAlgoTrig = cms.bool( True ),
    EnableL1GtTriggerMaskTechTrig = cms.bool( True ),
    EnableL1GtTriggerMaskVetoTechTrig = cms.bool( True )      
)