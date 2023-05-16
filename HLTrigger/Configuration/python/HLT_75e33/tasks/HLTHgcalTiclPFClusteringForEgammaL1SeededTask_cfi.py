import FWCore.ParameterSet.Config as cms

from ..modules.HGCalRecHitL1Seeded_cfi import *
from ..modules.HGCalUncalibRecHitL1Seeded_cfi import *
from ..modules.filteredLayerClustersCLUE3DHighL1Seeded_cfi import *
from ..modules.hgcalDigisL1Seeded_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalLayerClustersEEL1Seeded_cfi import *
from ..modules.hgcalLayerClustersHSiL1Seeded_cfi import *
from ..modules.hgcalLayerClustersHSciL1Seeded_cfi import *
from ..modules.hgcalMergeLayerClustersL1Seeded_cfi import *
from ..modules.hltL1TEGammaHGCFilteredCollectionProducer_cfi import *
from ..modules.hltRechitInRegionsHGCAL_cfi import *
from ..modules.particleFlowClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.particleFlowRecHitHGCL1Seeded_cfi import *
from ..modules.particleFlowSuperClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.ticlLayerTileProducerL1Seeded_cfi import *
from ..modules.ticlSeedingL1_cfi import *
from ..modules.ticlTrackstersCLUE3DHighL1Seeded_cfi import *
from ..tasks.HLTBeamSpotTask_cfi import *


HLTHgcalTiclPFClusteringForEgammaL1SeededTask = cms.Task(
    HGCalRecHitL1Seeded,
    HGCalUncalibRecHitL1Seeded,
    HLTBeamSpotTask,
    filteredLayerClustersCLUE3DHighL1Seeded,
    hgcalDigis,
    hgcalDigisL1Seeded,
    hgcalLayerClustersEEL1Seeded,
    hgcalLayerClustersHSciL1Seeded,
    hgcalLayerClustersHSiL1Seeded,
    hgcalMergeLayerClustersL1Seeded,
    hltL1TEGammaHGCFilteredCollectionProducer,
    hltRechitInRegionsHGCAL,
    particleFlowClusterHGCalFromTICLL1Seeded,
    particleFlowRecHitHGCL1Seeded,
    particleFlowSuperClusterHGCalFromTICLL1Seeded,
    ticlLayerTileProducerL1Seeded,
    ticlSeedingL1,
    ticlTrackstersCLUE3DHighL1Seeded
)
