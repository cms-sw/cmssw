import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersEMForEgamma_cfi import *
from ..modules.filteredLayerClustersHADForEgamma_cfi import *
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
from ..modules.ticlMultiClustersFromTrackstersEMForEgamma_cfi import *
from ..modules.ticlMultiClustersFromTrackstersHADForEgamma_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersEMForEgamma_cfi import *
from ..modules.ticlTrackstersHADForEgamma_cfi import *

HLTHgcalTiclPFClusteringForEgammaUnseededTask = cms.Task(
    HGCalRecHit,
    HGCalUncalibRecHit,
    filteredLayerClustersEMForEgamma,
    filteredLayerClustersHADForEgamma,
    hgcalDigis,
    hgcalLayerClusters,
    offlineBeamSpot,
    particleFlowClusterHGCalFromTICLHAD,
    particleFlowClusterHGCalFromTICLUnseeded,
    particleFlowRecHitHGC,
    particleFlowSuperClusterHGCalFromTICLUnseeded,
    ticlLayerTileProducer,
    ticlMultiClustersFromTrackstersEMForEgamma,
    ticlMultiClustersFromTrackstersHADForEgamma,
    ticlSeedingGlobal,
    ticlTrackstersEMForEgamma,
    ticlTrackstersHADForEgamma
)
