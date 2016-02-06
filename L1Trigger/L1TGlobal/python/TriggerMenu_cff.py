import FWCore.ParameterSet.Config as cms

L1TUtmTriggerMenuRcdSource = cms.ESSource("EmptyESSource",
    recordName = cms.string('L1TUtmTriggerMenuRcd'),
    iovIsRunNotTime = cms.bool(True),
    firstValid = cms.vuint32(1)
)

TriggerMenu = cms.ESProducer("L1TUtmTriggerMenuESProducer")
