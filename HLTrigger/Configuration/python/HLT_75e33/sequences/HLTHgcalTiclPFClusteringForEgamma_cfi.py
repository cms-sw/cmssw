import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalLayerClustersEE_cfi import *
from ..modules.hgcalLayerClustersHSci_cfi import *
from ..modules.hgcalLayerClustersHSi_cfi import *
from ..modules.hgcalMergeLayerClusters_cfi import *
from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
from ..modules.particleFlowClusterHGCal_cfi import *
from ..modules.particleFlowRecHitHGC_cfi import *
from ..modules.particleFlowSuperClusterHGCal_cfi import *
from ..modules.ticlLayerTileProducer_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersCLUE3DHigh_cfi import *

HLTHgcalTiclPFClusteringForEgamma = cms.Sequence((hgcalDigis+HGCalUncalibRecHit+HGCalRecHit+particleFlowRecHitHGC+hgcalLayerClustersEE+hgcalLayerClustersHSci+hgcalLayerClustersHSi+hgcalMergeLayerClusters+filteredLayerClustersCLUE3DHigh+ticlSeedingGlobal+ticlLayerTileProducer+ticlTrackstersCLUE3DHigh+particleFlowClusterHGCal+particleFlowSuperClusterHGCal))
