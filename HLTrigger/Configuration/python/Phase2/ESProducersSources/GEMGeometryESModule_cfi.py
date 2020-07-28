import FWCore.ParameterSet.Config as cms

GEMGeometryESModule = cms.ESProducer("GEMGeometryESModule",
    alignmentsLabel = cms.string(''),
    applyAlignment = cms.bool(False),
    useDDD = cms.bool(True)
)
