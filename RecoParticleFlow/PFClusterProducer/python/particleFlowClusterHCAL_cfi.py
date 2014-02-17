import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterCleaners_cfi \
     import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterSeeders_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterizers_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterPositionCalculators_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterEnergyCorrectors_cfi import *

particleFlowClusterHCAL = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHCAL"),
    recHitCleaners = cms.VPSet(rbxAndHPDCleaner),
    seedFinder = localMaxSeeds_HCAL,
    topoClusterBuilder = topoClusterizer_HCAL,
    pfClusterBuilder = pfClusterizer_HCAL,
    )

