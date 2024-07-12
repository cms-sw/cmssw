import FWCore.ParameterSet.Config as cms

from ..modules.hgcalLayerClustersEE_cfi import *
from ..modules.hgcalLayerClustersHSci_cfi import *
from ..modules.hgcalLayerClustersHSi_cfi import *
from ..modules.hgcalMergeLayerClusters_cfi import *
from ..modules.HGCalRecHit_cfi import *
from ..modules.HGCalUncalibRecHit_cfi import *
# Heterogeneous HGCAL EE layer clusters
from ..modules.hltHgcalSoARecHitsProducer_cfi import *
from ..modules.hltHgcalSoARecHitsLayerClustersProducer_cfi import *
from ..modules.hltHgcalSoALayerClustersProducer_cfi import *
from ..modules.hltHgcalLayerClustersFromSoAProducer_cfi import *

hgcalLocalRecoSequence = cms.Sequence(
        HGCalUncalibRecHit+
        HGCalRecHit+
        hgcalLayerClustersEE+
        hgcalLayerClustersHSci+
        hgcalLayerClustersHSi+
        hgcalMergeLayerClusters)

_hgcalLocalRecoSequence_heterogeneous = cms.Sequence(
        HGCalUncalibRecHit+
        HGCalRecHit+
        hltHgcalSoARecHitsProducer+
        hltHgcalSoARecHitsLayerClustersProducer+
        hltHgcalSoALayerClustersProducer+
        hltHgCalLayerClustersFromSoAProducer+
        hgcalLayerClustersHSci+
        hgcalLayerClustersHSi+
        hgcalMergeLayerClusters)

from Configuration.ProcessModifiers.alpaka_cff import alpaka
alpaka.toReplaceWith(hgcalLocalRecoSequence, _hgcalLocalRecoSequence_heterogeneous)
alpaka.toModify(hgcalMergeLayerClusters,
        layerClustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer"),
        time_layerclustersEE = cms.InputTag("hltHgCalLayerClustersFromSoAProducer", "timeLayerCluster"))
