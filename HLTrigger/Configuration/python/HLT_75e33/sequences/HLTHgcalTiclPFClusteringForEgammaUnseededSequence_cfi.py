import FWCore.ParameterSet.Config as cms

from ..modules.hltFilteredLayerClustersCLUE3DHigh_cfi import *
from ..modules.hltHgcalDigis_cfi import *
from ..modules.hltHgcalLayerClustersEE_cfi import *
from ..modules.hltHgcalLayerClustersHSci_cfi import *
from ..modules.hltHgcalLayerClustersHSi_cfi import *
from ..modules.hltHgcalMergeLayerClusters_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
from ..modules.hltParticleFlowClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.hltParticleFlowRecHitHGC_cfi import *
from ..modules.hltParticleFlowSuperClusterHGCalFromTICLUnseeded_cfi import *
from ..modules.hltTiclLayerTileProducer_cfi import *
from ..modules.hltTiclSeedingGlobal_cfi import *
from ..modules.hltTiclTrackstersCLUE3DHigh_cfi import *
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalLayerClustersFromSoAProducer_cfi import *

HLTHgcalTiclPFClusteringForEgammaUnseededSequence = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalLayerClustersEE+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltParticleFlowClusterHGCalFromTICLUnseeded+hltParticleFlowSuperClusterHGCalFromTICLUnseeded)

_HLTHgcalTiclPFClusteringForEgammaUnseededSequence_heterogeneous = cms.Sequence(hltHgcalDigis+hltHGCalUncalibRecHit+hltHGCalRecHit+hltParticleFlowRecHitHGC+hltHgcalSoARecHitsProducer+hltHgcalSoARecHitsLayerClustersProducer+hltHgcalSoALayerClustersProducer+hltHgCalLayerClustersFromSoAProducer+hltHgcalLayerClustersHSci+hltHgcalLayerClustersHSi+hltHgcalMergeLayerClusters+hltFilteredLayerClustersCLUE3DHigh+hltTiclSeedingGlobal+hltTiclLayerTileProducer+hltTiclTrackstersCLUE3DHigh+hltParticleFlowClusterHGCalFromTICLUnseeded+hltParticleFlowSuperClusterHGCalFromTICLUnseeded)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(HLTHgcalTiclPFClusteringForEgammaUnseededSequence, _HLTHgcalTiclPFClusteringForEgammaUnseededSequence_heterogeneous)
alpaka.toModify(hltHgcalMergeLayerClusters,
        layerClustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer"),
        time_layerclustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer", "timeLayerCluster"))
