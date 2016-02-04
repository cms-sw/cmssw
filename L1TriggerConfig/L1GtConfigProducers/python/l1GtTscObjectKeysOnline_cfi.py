# cfi file to produce the L1 GT keys 

import FWCore.ParameterSet.Config as cms
l1GtTscObjectKeysOnline = cms.ESProducer("L1GtTscObjectKeysOnlineProd",
    onlineAuthentication = cms.string('.'),
    subsystemLabel = cms.string('GT'),
    onlineDB = cms.string('oracle://CMS_OMDS_LB/CMS_TRG_R'),
    #
    EnableL1GtParameters = cms.bool( True ),
    EnableL1GtTriggerMenu = cms.bool( True ),
    EnableL1GtPsbSetup = cms.bool( True )
    
)