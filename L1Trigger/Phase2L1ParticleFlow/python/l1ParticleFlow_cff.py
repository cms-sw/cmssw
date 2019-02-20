import FWCore.ParameterSet.Config as cms

from L1Trigger.L1CaloTrigger.l1EGammaCrystalsProducer_cfi import *

from L1Trigger.Phase2L1ParticleFlow.pfTracksFromL1Tracks_cfi import *
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromHGC3DClustersEM_cfi import *
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromL1EGClusters_cfi import *
from L1Trigger.Phase2L1ParticleFlow.pfClustersFromCombinedCalo_cfi import *
from L1Trigger.Phase2L1ParticleFlow.l1pfProducer_cfi import *

l1PuppiForMET = cms.EDFilter("L1TPFCandSelector", 
    src = cms.InputTag("l1pfProducer:Puppi"), 
    cut = cms.string("charge != 0 ||"+
                     "abs(eta) < 1.5 ||"+
                     "(pt > 20 && abs(eta) < 2.5) ||"+
                     "(pt > 40 && 2.5 <= abs(eta) <= 2.85) ||"+
                     "(pt > 30 && abs(eta) > 3.0)")
)

l1ParticleFlow = cms.Sequence(
    l1EGammaCrystalsProducer + 
    pfTracksFromL1Tracks +
    pfClustersFromHGC3DClustersEM +
    pfClustersFromL1EGClusters +
    pfClustersFromCombinedCalo +
    l1pfProducer +
    l1PuppiForMET
)
