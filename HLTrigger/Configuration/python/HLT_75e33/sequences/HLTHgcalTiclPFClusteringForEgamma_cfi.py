import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalLayerClustersEE_cfi import *
from ..modules.hltHgcalLayerClustersHSci_cfi import *
from ..modules.hltHgcalLayerClustersHSi_cfi import *
from ..modules.hltMergeLayerClusters_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
from ..modules.hltParticleFlowClusterHGCal_cfi import *
from ..modules.hltParticleFlowRecHitHGC_cfi import *
from ..modules.hltParticleFlowSuperClusterHGCal_cfi import *
from ..modules.hltTiclLayerTileProducer_cfi import *
from ..modules.hltTiclSeedingGlobal_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHigh_cfi import *
# Barrel layer clusters
from ..modules.hltBarrelLayerClustersEB_cfi import *
from ..modules.hltBarrelLayerClustersHB_cfi import *
HLTHgcalTiclPFClusteringForEgamma = cms.Sequence((hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalLayerClustersEE+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltParticleFlowClusterHGCal+hltParticleFlowSuperClusterHGCal))

HLTHgcalTiclPFClusteringForEgamma_barrel = cms.Sequence((
  hltHgcalDigis+
  hltHGCalUncalibRecHit+
  hltHGCalRecHit+
  hltParticleFlowRecHitHGC+
  hltHgcalLayerClustersEE+
  hltHgcalLayerClustersHSci+
  hltHgcalLayerClustersHSi+
  hltBarrelLayerClustersEB+
  hltBarrelLayerClustersHB+
  hltMergeLayerClusters+
  hltFilteredLayerClustersCLUE3DHigh+
  hltTiclSeedingGlobal+
  hltTiclLayerTileProducer+
  hltTiclTrackstersCLUE3DHigh+
  hltParticleFlowClusterHGCal+
  hltParticleFlowSuperClusterHGCal
))
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(HLTHgcalTiclPFClusteringForEgamma, HLTHgcalTiclPFClusteringForEgamma_barrel)

