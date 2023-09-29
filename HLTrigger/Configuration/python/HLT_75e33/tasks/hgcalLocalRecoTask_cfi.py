import FWCore.ParameterSet.Config as cms

from ..modules.hgcalLayerClustersEE_cfi import *
from ..modules.hgcalLayerClustersHSi_cfi import *
from ..modules.hgcalLayerClustersHSci_cfi import *
from ..modules.hgcalMergeLayerClusters_cfi import *
from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
from ..modules.particleFlowClusterHGCal_cfi import *
from ..modules.particleFlowRecHitHGC_cfi import *

hgcalLocalRecoTask = cms.Task(
    HGCalRecHit,
    HGCalUncalibRecHit,
    hgcalLayerClustersEE,
    hgcalLayerClustersHSi,
    hgcalLayerClustersHSci,
    hgcalMergeLayerClusters,
    particleFlowClusterHGCal,
    particleFlowRecHitHGC
)
