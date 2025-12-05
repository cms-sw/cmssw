import FWCore.ParameterSet.Config as cms

hcalPulseShapesESProd = cms.ESProducer('HcalPulseShapesEP',
    productLabel = cms.string('HcalDataShapes'),
    pulseShapeLength = cms.uint32(250),
    globalTimeShift = cms.int32(0),
    pulseDumpFile = cms.untracked.string(''),
    dumpPrecision = cms.untracked.uint32(0)
)
