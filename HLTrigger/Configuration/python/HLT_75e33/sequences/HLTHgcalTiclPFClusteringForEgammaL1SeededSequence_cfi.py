import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersCLUE3DHighL1Seeded_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalDigisL1Seeded_cfi import *
from ..modules.hgcalLayerClustersEEL1Seeded_cfi import *
from ..modules.hgcalLayerClustersHSciL1Seeded_cfi import *
from ..modules.hgcalLayerClustersHSiL1Seeded_cfi import *
from ..modules.hgcalMergeLayerClustersL1Seeded_cfi import *
from ..modules.HGCalRecHitL1Seeded_cfi import *
from ..modules.HGCalUncalibRecHitL1Seeded_cfi import *
from ..modules.hltL1TEGammaHGCFilteredCollectionProducer_cfi import *
from ..modules.hltRechitInRegionsHGCAL_cfi import *
from ..modules.particleFlowClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.particleFlowRecHitHGCL1Seeded_cfi import *
from ..modules.particleFlowSuperClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.ticlLayerTileProducerL1Seeded_cfi import *
from ..modules.ticlSeedingL1_cfi import *
from ..modules.ticlTrackstersCLUE3DHighL1Seeded_cfi import *

HLTHgcalTiclPFClusteringForEgammaL1SeededSequence = cms.Sequence(hgcalDigis+hltL1TEGammaHGCFilteredCollectionProducer+hgcalDigisL1Seeded+HGCalUncalibRecHitL1Seeded+HGCalRecHitL1Seeded+particleFlowRecHitHGCL1Seeded+hltRechitInRegionsHGCAL+hgcalLayerClustersEEL1Seeded+hgcalLayerClustersHSciL1Seeded+hgcalLayerClustersHSiL1Seeded+hgcalMergeLayerClustersL1Seeded+filteredLayerClustersCLUE3DHighL1Seeded+ticlSeedingL1+ticlLayerTileProducerL1Seeded+ticlTrackstersCLUE3DHighL1Seeded+particleFlowClusterHGCalFromTICLL1Seeded+particleFlowSuperClusterHGCalFromTICLL1Seeded)
