import FWCore.ParameterSet.Config as cms

mtdNumberingGeometry = cms.ESProducer("MTDGeometricTimingDetESModule",
    appendToDataLabel = cms.string(''),
    fromDD4hep = cms.bool(False),
    fromDDD = cms.bool(True)
)
