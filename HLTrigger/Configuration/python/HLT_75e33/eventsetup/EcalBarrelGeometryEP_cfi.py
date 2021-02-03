import FWCore.ParameterSet.Config as cms

EcalBarrelGeometryEP = cms.ESProducer("EcalBarrelGeometryEP",
    applyAlignment = cms.bool(False)
)
