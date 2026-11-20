import FWCore.ParameterSet.Config as cms

from .hltMergeLayerClusters_cfi import hltMergeLayerClusters as _hltMergeLayerClusters

_layerClusters = [
    "hltHgcalLayerClustersEEMuonSeeded",
    "hltHgcalLayerClustersHSciMuonSeeded",
    "hltHgcalLayerClustersHSiMuonSeeded"
]
# barrel hadronic layer clusters are taken from the full HCAL reconstruction
_barrelLayerClusters = [
    "hltBarrelLayerClustersEBMuonSeeded",
    "hltBarrelLayerClustersHB"
]

# as in the e/gamma L1-seeded chain, the EE layer clusters come from the CPU producer also under alpaka
hltMergeLayerClustersMuonSeeded = _hltMergeLayerClusters.clone(
    layerClusters = _layerClusters,
    time_layerclusters = [x + ":timeLayerCluster" for x in _layerClusters]
)

from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toModify(hltMergeLayerClustersMuonSeeded,
    layerClusters = _layerClusters + _barrelLayerClusters,
    time_layerclusters = [x + ":timeLayerCluster" for x in _layerClusters + _barrelLayerClusters]
)
