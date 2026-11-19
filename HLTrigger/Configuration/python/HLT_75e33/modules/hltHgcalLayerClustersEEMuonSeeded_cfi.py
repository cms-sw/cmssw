import FWCore.ParameterSet.Config as cms

from .hltHgcalLayerClustersEE_cfi import hltHgcalLayerClustersEE as _hltHgcalLayerClustersEE

hltHgcalLayerClustersEEMuonSeeded = _hltHgcalLayerClustersEE.clone(
    recHits = "hltHGCalRecHitMuonSeeded:HGCEERecHits"
)
