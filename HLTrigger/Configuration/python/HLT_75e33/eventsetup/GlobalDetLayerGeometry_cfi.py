import FWCore.ParameterSet.Config as cms

GlobalDetLayerGeometry = cms.ESProducer("GlobalDetLayerGeometryESProducer",
    ComponentName = cms.string('GlobalDetLayerGeometry')
)
