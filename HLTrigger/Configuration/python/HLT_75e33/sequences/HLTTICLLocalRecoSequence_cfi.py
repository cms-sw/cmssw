import FWCore.ParameterSet.Config as cms

from ..modules.hltHgcalLayerClustersEE_cfi import *
from ..modules.hltHgcalLayerClustersHSci_cfi import *
from ..modules.hltHgcalLayerClustersHSi_cfi import *
from ..modules.hltMergeLayerClusters_cfi import *
from ..modules.hltHGCalRecHit_cfi import *
from ..modules.hltHGCalUncalibRecHit_cfi import *
# Heterogeneous HGCAL EE layer clusters
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalLayerClustersFromSoAProducer_cfi import *
# Barrel layer clusters
from ..modules.hltBarrelLayerClustersEB_cfi import *
from ..modules.hltBarrelLayerClustersHB_cfi import *

HLTTICLLocalRecoSequence = cms.Sequence(
        hltHGCalUncalibRecHit+
        hltHGCalRecHit+
        hltHgcalLayerClustersEE+
        hltHgcalLayerClustersHSci+
        hltHgcalLayerClustersHSi+
        hltMergeLayerClusters)

_HLTTICLLocalRecoSequence_heterogeneous = cms.Sequence(
        hltHGCalUncalibRecHit+
        hltHGCalRecHit+
        hltHgcalSoARecHitsProducer+
        hltHgcalSoARecHitsLayerClustersProducer+
        hltHgcalSoALayerClustersProducer+
        hltHgCalLayerClustersFromSoAProducer+
        hltHgcalLayerClustersHSci+
        hltHgcalLayerClustersHSi+
        hltBarrelLayerClustersEB+
        hltBarrelLayerClustersHB+
        hltMergeLayerClusters)

layerClusters = cms.VInputTag("hltHgCalLayerClustersFromSoAProducer", "hltHgcalLayerClustersHSci", "hltHgcalLayerClustersHSi")
time_layerclusters = cms.VInputTag("hltHgCalLayerClustersFromSoAProducer:timeLayerCluster", "hltHgcalLayerClustersHSci:timeLayerCluster", "hltHgcalLayerClustersHSi:timeLayerCluster")
from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(HLTTICLLocalRecoSequence, _HLTTICLLocalRecoSequence_heterogeneous)
alpaka.toModify(hltMergeLayerClusters,
        layerClusters = layerClusters,
        time_layerclusters = time_layerclusters)


_HLTTICLLocalRecoSequence_withBarrel = cms.Sequence(
        hltHGCalUncalibRecHit+
        hltHGCalRecHit+
        hltHgcalLayerClustersEE+
        hltHgcalLayerClustersHSci+
        hltHgcalLayerClustersHSi+
        hltBarrelLayerClustersEB+
        hltBarrelLayerClustersHB+
        hltMergeLayerClusters
)

_HLTTICLLocalRecoSequence_heterogeneous_withBarrel = cms.Sequence(
        hltHGCalUncalibRecHit+
        hltHGCalRecHit+
        hltHgcalSoARecHitsProducer+
        hltHgcalSoARecHitsLayerClustersProducer+
        hltHgcalSoALayerClustersProducer+
        hltHgCalLayerClustersFromSoAProducer+
        hltHgcalLayerClustersHSci+
        hltHgcalLayerClustersHSi+
        hltBarrelLayerClustersEB+
        hltBarrelLayerClustersHB+
        hltMergeLayerClusters
)

ayerClusters = cms.VInputTag("hltHgCalLayerClustersFromSoAProducer", "hltHgcalLayerClustersHSci", "hltHgcalLayerClustersHSi", "hltBarrelLayerClustersEB", "hltBarrelLayerClustersHB")
time_layerclusters = cms.VInputTag("hltHgCalLayerClustersFromSoAProducer:timeLayerCluster", "hltHgcalLayerClustersHSci:timeLayerCluster", "hltHgcalLayerClustersHSi:timeLayerCluster", "hltBarrelLayerClustersEB:timeLayerCluster", "hltBarrelLayerClustersHB:timeLayerCluster")
from Configuration.ProcessModifiers.ticl_barrel_cff import ticl_barrel
ticl_barrel.toReplaceWith(HLTTICLLocalRecoSequence, _HLTTICLLocalRecoSequence_withBarrel)
(ticl_barrel & alpaka).toReplaceWith(HLTTICLLocalRecoSequence, _HLTTICLLocalRecoSequence_heterogeneous_withBarrel)
(ticl_barrel & alpaka).toModify(hltMergeLayerClusters,
        layerClusters = layerClusters,
        time_layerclusters = time_layerclusters
)
