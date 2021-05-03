import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersEML1Seeded_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalDigisL1Seeded_cfi import *
from ..modules.hgcalLayerClustersL1Seeded_cfi import *
from ..modules.HGCalRecHitL1Seeded_cfi import *
from ..modules.HGCalUncalibRecHitL1Seeded_cfi import *
from ..modules.hltL1TEGammaHGCFilteredCollectionProducer_cfi import *
from ..modules.hltRechitInRegionsHGCAL_cfi import *
from ..modules.offlineBeamSpot_cfi import *
from ..modules.particleFlowClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.particleFlowRecHitHGCL1Seeded_cfi import *
from ..modules.particleFlowSuperClusterHGCalFromTICLL1Seeded_cfi import *
from ..modules.ticlLayerTileProducerL1Seeded_cfi import *
from ..modules.ticlMultiClustersFromTrackstersMergeL1Seeded_cfi import *
from ..modules.ticlSeedingL1_cfi import *
from ..modules.ticlTrackstersEML1_cfi import *

HLTHgcalTiclPFClusteringForEgammaL1SeededTask = cms.Task(HGCalRecHitL1Seeded, HGCalUncalibRecHitL1Seeded, filteredLayerClustersEML1Seeded, hgcalDigis, hgcalDigisL1Seeded, hgcalLayerClustersL1Seeded, hltL1TEGammaHGCFilteredCollectionProducer, hltRechitInRegionsHGCAL, offlineBeamSpot, particleFlowClusterHGCalFromTICLL1Seeded, particleFlowRecHitHGCL1Seeded, particleFlowSuperClusterHGCalFromTICLL1Seeded, ticlLayerTileProducerL1Seeded, ticlMultiClustersFromTrackstersMergeL1Seeded, ticlSeedingL1, ticlTrackstersEML1)
