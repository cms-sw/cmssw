import FWCore.ParameterSet.Config as cms

L1TUtmTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TUtmTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

TriggerMenu = cms.ESProducer("L1TUtmTriggerMenuESProducer",

    # XML file for Global Trigger menu (menu.xml) 
    L1TriggerMenuFile = cms.string('L1Menu_Collisions2015_25nsStage1_v7_uGT.xml'),
    
)
