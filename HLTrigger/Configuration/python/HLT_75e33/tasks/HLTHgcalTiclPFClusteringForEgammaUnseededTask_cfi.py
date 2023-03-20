import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalLayerClusters_cfi import *
from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
from ..modules.offlineBeamSpot_cfi import *
from ..modules.particleFlowClusterHGCalFromTICLHAD_cfi import *
from ..modules.particleFlowClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.particleFlowRecHitHGC_cfi import *
from ..modules.particleFlowSuperClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.ticlLayerTileProducer_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersCLUE3DHigh_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalMergeLayerClusters_cfi import hgcalMergeLayerClusters


HLTHgcalTiclPFClusteringForEgammaUnseededTask = cms.Task(
    HGCalRecHit,
    HGCalUncalibRecHit,
    filteredLayerClustersCLUE3DHigh,
    hgcalDigis,
    hgcalLayerClustersEE,
    hgcalLayerClustersHSi,
    hgcalLayerClustersHSci,
    hgcalMergeLayerClusters,
    offlineBeamSpot,
    particleFlowClusterHGCalFromTICLHAD,
    particleFlowClusterHGCalFromTICLUnseeded,
    particleFlowRecHitHGC,
    particleFlowSuperClusterHGCalFromTICLUnseeded,
    ticlLayerTileProducer,
    ticlSeedingGlobal,
    ticlTrackstersCLUE3DHigh
)
