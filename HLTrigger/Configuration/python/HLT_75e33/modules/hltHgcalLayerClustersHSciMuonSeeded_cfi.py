import FWCore.ParameterSet.Config as cms

from .hltHgcalLayerClustersHSci_cfi import hltHgcalLayerClustersHSci as _hltHgcalLayerClustersHSci

hltHgcalLayerClustersHSciMuonSeeded = _hltHgcalLayerClustersHSci.clone(
    recHits = "hltHGCalRecHitMuonSeeded:HGCHEBRecHits"
)
