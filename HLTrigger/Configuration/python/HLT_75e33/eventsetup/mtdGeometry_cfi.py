import FWCore.ParameterSet.Config as cms

mtdGeometry = cms.ESProducer("MTDDigiGeometryESModule",
    alignmentsLabel = cms.string(''),
    appendToDataLabel = cms.string(''),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True)
)
