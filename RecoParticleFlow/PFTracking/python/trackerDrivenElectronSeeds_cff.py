import TrackingTools.GsfTracking.GsfElectronMaterialEffects_cfi

ElectronMaterialEffects_forPreId = TrackingTools.GsfTracking.GsfElectronMaterialEffects_cfi.ElectronMaterialEffects.clone(
    ComponentName = 'ElectronMaterialEffects_forPreId',
    BetheHeitlerParametrization = 'BetheHeitler_cdfmom_nC3_O5.par'
)
from RecoParticleFlow.PFTracking.trackerDrivenElectronSeeds_cfi import *
CloseComponentsMerger_forPreId = cms.ESProducer("CloseComponentsMergerESProducer5D",
    ComponentName = cms.string('CloseComponentsMerger_forPreId'),
    MaxComponents = cms.int32(4),
    DistanceMeasure = cms.string('KullbackLeiblerDistance5D')
)

GsfTrajectoryFitter_forPreId = cms.ESProducer("GsfTrajectoryFitterESProducer",
    Merger = cms.string('CloseComponentsMerger_forPreId'),
    ComponentName = cms.string('GsfTrajectoryFitter_forPreId'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects_forPreId'),
    GeometricalPropagator = cms.string('fwdAnalyticalPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)

GsfTrajectorySmoother_forPreId = cms.ESProducer("GsfTrajectorySmootherESProducer",
    Merger = cms.string('CloseComponentsMerger_forPreId'),
    ComponentName = cms.string('GsfTrajectorySmoother_forPreId'),
    MaterialEffectsUpdator = cms.string('ElectronMaterialEffects_forPreId'),
    ErrorRescaling = cms.double(100.0),
    GeometricalPropagator = cms.string('bwdAnalyticalPropagator'),
    RecoGeometry = cms.string('GlobalDetLayerGeometry')
)

