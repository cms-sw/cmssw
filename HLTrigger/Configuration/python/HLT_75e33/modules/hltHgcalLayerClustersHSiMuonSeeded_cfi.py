import FWCore.ParameterSet.Config as cms

from .hltHgcalLayerClustersHSi_cfi import hltHgcalLayerClustersHSi as _hltHgcalLayerClustersHSi

hltHgcalLayerClustersHSiMuonSeeded = _hltHgcalLayerClustersHSi.clone(
    recHits = "hltHGCalRecHitMuonSeeded:HGCHEFRecHits"
)
