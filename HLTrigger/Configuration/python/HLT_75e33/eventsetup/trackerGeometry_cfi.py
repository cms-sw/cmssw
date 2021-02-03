import FWCore.ParameterSet.Config as cms

trackerGeometry = cms.ESProducer("TrackerDigiGeometryESModule",
    alignmentsLabel = cms.string(''),
    appendToDataLabel = cms.string(''),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True)
)
