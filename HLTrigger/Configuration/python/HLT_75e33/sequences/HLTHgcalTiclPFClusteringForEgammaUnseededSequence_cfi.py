import FWCore.ParameterSet.Config as cms

from ..modules.filteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hgcalDigis_cfi import *
from ..modules.hgcalLayerClustersEE_cfi import *
from ..modules.hgcalLayerClustersHSci_cfi import *
from ..modules.hgcalLayerClustersHSi_cfi import *
from ..modules.hgcalMergeLayerClusters_cfi import *
from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
from ..modules.particleFlowClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.particleFlowRecHitHGC_cfi import *
from ..modules.particleFlowSuperClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.ticlLayerTileProducer_cfi import *
from ..modules.ticlSeedingGlobal_cfi import *
from ..modules.ticlTrackstersCLUE3DHigh_cfi import *
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalLayerClustersFromSoAProducer_cfi import *

HLTHgcalTiclPFClusteringForEgammaUnseededSequence = cms.Sequence(hgcalDigis+HGCalUncalibRecHit+HGCalRecHit+particleFlowRecHitHGC+hgcalLayerClustersEE+hgcalLayerClustersHSci+hgcalLayerClustersHSi+hgcalMergeLayerClusters+filteredLayerClustersCLUE3DHigh+ticlSeedingGlobal+ticlLayerTileProducer+ticlTrackstersCLUE3DHigh+particleFlowClusterHGCalFromTICLUnseeded+particleFlowSuperClusterHGCalFromTICLUnseeded)

_HLTHgcalTiclPFClusteringForEgammaUnseededSequence_heterogeneous = cms.Sequence(hgcalDigis+HGCalUncalibRecHit+HGCalRecHit+particleFlowRecHitHGC+hltHgcalSoARecHitsProducer+hltHgcalSoARecHitsLayerClustersProducer+hltHgcalSoALayerClustersProducer+hltHgCalLayerClustersFromSoAProducer+hgcalLayerClustersHSci+hgcalLayerClustersHSi+hgcalMergeLayerClusters+filteredLayerClustersCLUE3DHigh+ticlSeedingGlobal+ticlLayerTileProducer+ticlTrackstersCLUE3DHigh+particleFlowClusterHGCalFromTICLUnseeded+particleFlowSuperClusterHGCalFromTICLUnseeded)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(HLTHgcalTiclPFClusteringForEgammaUnseededSequence, _HLTHgcalTiclPFClusteringForEgammaUnseededSequence_heterogeneous)
alpaka.toModify(hgcalMergeLayerClusters,
        layerClustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer"),
        time_layerclustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer", "timeLayerCluster"))
