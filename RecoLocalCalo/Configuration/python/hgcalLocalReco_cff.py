import FWCore.ParameterSet.Config as cms

from RecoLocalCalo.HGCalRecProducers.HGCalUncalibRecHit_cfi import *
from RecoLocalCalo.HGCalRecProducers.HGCalRecHit_cfi import *

# patch particle flow clusters for HGC into local reco sequence
# (for now until global reco is going with some sort of clustering)
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHGC_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHGC_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalMultiClusters_cfi import *
from RecoLocalCalo.HGCalRecProducers.hgcalLayerClusters_cff import hgcalLayerClusters

hgcalLocalRecoTask = cms.Task( HGCalUncalibRecHit,
                                       HGCalRecHit,
                                       hgcalLayerClusters,
                                       hgcalMultiClusters,
                                       particleFlowRecHitHGC,
                                       particleFlowClusterHGCal,
                                       particleFlowClusterHGCalFromMultiCl )
hgcalLocalRecoSequence = cms.Sequence(hgcalLocalRecoTask)
