import FWCore.ParameterSet.Config as cms

idealForDigiMTDGeometry = cms.ESProducer("MTDDigiGeometryESModule",
    alignmentsLabel = cms.string('fakeForIdeal'),
    appendToDataLabel = cms.string('idealForDigi'),
    applyAlignment = cms.bool(False),
    fromDDD = cms.bool(True)
)
