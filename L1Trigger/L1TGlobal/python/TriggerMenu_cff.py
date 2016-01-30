#
# WARNING: This file is in the L1T configuration critical path.
#
# All changes must be explicitly discussed with the L1T offline coordinator.
#
import FWCore.ParameterSet.Config as cms

L1TUtmTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TUtmTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

TriggerMenu = cms.ESProducer("L1TUtmTriggerMenuESProducer",
    # XML file for Global Trigger menu (menu.xml) 
    L1TriggerMenuFile = cms.string('Overide_This_Value.xml'),
)
