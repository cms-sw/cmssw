import FWCore.ParameterSet.Config as cms

from .hltBarrelLayerClustersEB_cfi import hltBarrelLayerClustersEB as _hltBarrelLayerClustersEB

hltBarrelLayerClustersEBMuonSeeded = _hltBarrelLayerClustersEB.clone(
    recHits = "hltParticleFlowRecHitECALMuonSeeded"
)
