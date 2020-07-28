import FWCore.ParameterSet.Config as cms

ME0GeometryESModule = cms.ESProducer("ME0GeometryESModule",
    use10EtaPart = cms.bool(True),
    useDDD = cms.bool(True)
)
