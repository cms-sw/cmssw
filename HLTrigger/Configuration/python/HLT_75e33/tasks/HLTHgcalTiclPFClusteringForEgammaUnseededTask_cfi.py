import FWCore.ParameterSet.Config as cms

from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
from ..modules.filteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalLayerClusters_cfi import *
from ..modules.particleFlowClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.particleFlowRecHitHGC_cfi import *
from ..modules.particleFlowSuperClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.ticlLayerTileProducer_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersCLUE3DHigh_cfi import *
from ..modules.hgcalMergeLayerClusters_cfi import hgcalMergeLayerClusters
from ..tasks.HLTBeamSpotTask_cfi import *

HLTHgcalTiclPFClusteringForEgammaUnseededTask = cms.Task(
    HGCalRecHit,
    HGCalUncalibRecHit,
    HLTBeamSpotTask,
    filteredLayerClustersCLUE3DHigh,
    hgcalDigis,
    hgcalLayerClustersEE,
    hgcalLayerClustersHSi,
    hgcalLayerClustersHSci,
    hgcalMergeLayerClusters,
    particleFlowClusterHGCalFromTICLUnseeded,
    particleFlowRecHitHGC,
    particleFlowSuperClusterHGCalFromTICLUnseeded,
    ticlLayerTileProducer,
    ticlSeedingGlobal,
    ticlTrackstersCLUE3DHigh
)
