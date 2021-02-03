import FWCore.ParameterSet.Config as cms

GsfTrajectoryFitter = cms.ESProducer("GsfTrajectoryFitterESProducer",
    ComponentName = cms.string('GsfTrajectoryFitter'),
    GeometricalPropagator = cms.string('fwdAnalyticalPropagator'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects'),
    Merger = cms.string('CloseComponentsMerger5D'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)
