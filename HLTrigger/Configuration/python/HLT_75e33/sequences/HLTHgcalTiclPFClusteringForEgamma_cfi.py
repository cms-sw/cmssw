import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalLayerClustersEE_cfi import *
from ..modules.hltHgcalLayerClustersHSci_cfi import *
from ..modules.hltHgcalLayerClustersHSi_cfi import *
from ..modules.hltHgcalMergeLayerClusters_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
from ..modules.hltParticleFlowClusterHGCal_cfi import *
from ..modules.hltParticleFlowRecHitHGC_cfi import *
from ..modules.hltParticleFlowSuperClusterHGCal_cfi import *
from ..modules.hltTiclLayerTileProducer_cfi import *
from ..modules.hltTiclSeedingGlobal_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHigh_cfi import *

HLTHgcalTiclPFClusteringForEgamma = cms.Sequence((hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalLayerClustersEE+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltParticleFlowClusterHGCal+hltParticleFlowSuperClusterHGCal))
