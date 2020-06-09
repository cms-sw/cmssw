import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

from SimCalorimetry.HGCalSimProducers.hgcHitAssociation_cfi import hgcRecHitMapProducer

# patch particle flow clusters for HGC into local reco sequence
# (for now until global reco is going with some sort of clustering)
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters, hgcalLayerClustersHFNose

hgcalLocalRecoTask = cms.Task( HGCalUncalibRecHit,
                                       HGCalRecHit,
                                       hgcRecHitMapProducer,
                                       hgcalLayerClusters,
                                       hgcalMultiClusters,
                                       particleFlowRecHitHGC,
                                       particleFlowClusterHGCal,
                                       particleFlowClusterHGCalFromMultiCl )

_hfnose_hgcalLocalRecoTask = hgcalLocalRecoTask.copy()
_hfnose_hgcalLocalRecoTask.add(hgcalLayerClustersHFNose)

from Configuration.Eras.Modifier_phase2_hfnose_cff import phase2_hfnose
phase2_hfnose.toReplaceWith(
    hgcalLocalRecoTask, _hfnose_hgcalLocalRecoTask )

hgcalLocalRecoSequence = cms.Sequence(hgcalLocalRecoTask)
