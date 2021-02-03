import FWCore.ParameterSet.Config as cms

GsfTrajectorySmoother = cms.ESProducer("GsfTrajectorySmootherESProducer",
    ComponentName = cms.string('GsfTrajectorySmoother'),
    ErrorRescaling = cms.double(100.0),
    GeometricalPropagator = cms.string('bwdAnalyticalPropagator'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects'),
    Merger = cms.string('CloseComponentsMerger5D'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)
