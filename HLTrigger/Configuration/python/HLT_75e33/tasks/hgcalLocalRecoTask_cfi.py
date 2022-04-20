import FWCore.ParameterSet.Config as cms

from ..modules.hgcalLayerClusters_cfi import *
from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
from ..modules.particleFlowClusterHGCal_cfi import *
#from ..modules.particleFlowClusterHGCalFromMultiCl_cfi import *
from ..modules.particleFlowRecHitHGC_cfi import *

hgcalLocalRecoTask = cms.Task(
    HGCalRecHit,
    HGCalUncalibRecHit,
    hgcalLayerClusters,
    particleFlowClusterHGCal,
#    particleFlowClusterHGCalFromMultiCl,
    particleFlowRecHitHGC
)
