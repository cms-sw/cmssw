import FWCore.ParameterSet.Config as cms

from RecoHGCal.TICL.filteredLayerClustersProducer_cfi import filteredLayerClustersProducer as _filteredLayerClustersProducer

filteredLayerClustersCLUE3DBarrel = _filteredLayerClustersProducer.clone(
    clusterFilter = "ClusterFilterByAlgo",
    algo_numer = [10, 11],
    iteration_label = "CLUE3DBarrel",
    max_layerid = 5
)  
