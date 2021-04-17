import FWCore.ParameterSet.Config as cms

import TrackingTools.MaterialEffects.MaterialPropagator_cfi
#PropagatorWithMaterialESProducer 
alongMomElePropagator = TrackingTools.MaterialEffects.MaterialPropagator_cfi.MaterialPropagator.clone(
    Mass          = 0.000511,
    ComponentName = 'alongMomElePropagator'
)
