import FWCore.ParameterSet.Config as cms

GsfTrajectoryFitter_forPreId = cms.ESProducer("GsfTrajectoryFitterESProducer",
    ComponentName = cms.string('GsfTrajectoryFitter_forPreId'),
    GeometricalPropagator = cms.string('fwdAnalyticalPropagator'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects_forPreId'),
    Merger = cms.string('CloseComponentsMerger_forPreId'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)
