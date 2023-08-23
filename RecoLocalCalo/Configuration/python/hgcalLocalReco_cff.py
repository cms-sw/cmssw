import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

from RecoLocalCalo.HGCalRecProducers.hgcalRecHitMapProducer_cfi import hgcalRecHitMapProducer

# patch particle flow clusters for HGC into local reco sequence
# (for now until global reco is going with some sort of clustering)
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClustersHFNose, hgcalLayerClustersEE, hgcalLayerClustersHSi, hgcalLayerClustersHSci, hgcalMergeLayerClusters

hgcalLocalRecoTask = cms.Task( HGCalUncalibRecHit,
                                       HGCalRecHit,
                                       hgcalRecHitMapProducer,
                                       hgcalLayerClustersEE,
                                       hgcalLayerClustersHSi,
                                       hgcalLayerClustersHSci,
                                       hgcalMergeLayerClusters,
                                       hgcalMultiClusters,
                                       particleFlowRecHitHGC,
                                       particleFlowClusterHGCal )

_hfnose_hgcalLocalRecoTask = hgcalLocalRecoTask.copy()
_hfnose_hgcalLocalRecoTask.add(hgcalLayerClustersHFNose)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toReplaceWith(
    hgcalLocalRecoTask, _hfnose_hgcalLocalRecoTask )

hgcalLocalRecoSequence = cms.Sequence(hgcalLocalRecoTask)
