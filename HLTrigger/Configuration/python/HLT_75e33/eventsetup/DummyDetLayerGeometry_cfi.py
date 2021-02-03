import FWCore.ParameterSet.Config as cms

DummyDetLayerGeometry = cms.ESProducer("DetLayerGeometryESProducer",
    ComponentName = cms.string('DummyDetLayerGeometry')
)
