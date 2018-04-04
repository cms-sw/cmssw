import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.l1EGammaCrystalsProducer_cfi import *

from L1Trigger.Phase2L1ParticleFlow.pfTracksFromL1Tracks_cfi import *
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromHGC3DClustersEM_cfi import *
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromL1EGClusters_cfi import *
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromCombinedCalo_cfi import *
from L1Trigger.Phase2L1ParticleFlow.l1pfProducer_cfi import *

l1ParticleFlow = cms.Sequence(
    l1EGammaCrystalsProducer + 
    pfTracksFromL1Tracks +
    pfClustersFromHGC3DClustersEM +
    pfClustersFromL1EGClusters +
    pfClustersFromCombinedCalo +
    l1pfProducer
)
