import FWCore.ParameterSet.Config as cms

GsfTrajectorySmoother_forPreId = cms.ESProducer("GsfTrajectorySmootherESProducer",
    ComponentName = cms.string('GsfTrajectorySmoother_forPreId'),
    ErrorRescaling = cms.double(100.0),
    GeometricalPropagator = cms.string('bwdAnalyticalPropagator'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects_forPreId'),
    Merger = cms.string('CloseComponentsMerger_forPreId'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)
