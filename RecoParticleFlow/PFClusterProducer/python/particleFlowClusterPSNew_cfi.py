import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterCleaners_cfi \
     import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterSeeders_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterizers_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterPositionCalculators_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterEnergyCorrectors_cfi import *

particleFlowClusterPSNew = cms.EDProducer(
    "PFClusterProducerNew",
    recHitsSource = cms.InputTag("particleFlowRecHitPS"),
    recHitCleaners = cms.VPSet(),
    seedFinder = localMaxSeeds_PS,
    topoClusterBuilder = topoClusterizer_PS,
    pfClusterBuilder = pfClusterizer_PS,
    )

