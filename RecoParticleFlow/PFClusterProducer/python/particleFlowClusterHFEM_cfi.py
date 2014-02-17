import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowClusterCleaners_cfi \
     import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterSeeders_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterizers_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterPositionCalculators_cfi import *

from RecoParticleFlow.PFClusterProducer.\
     particleFlowClusterEnergyCorrectors_cfi import *

particleFlowClusterHFEM = cms.EDProducer(
    "PFClusterProducer",
    recHitsSource = cms.InputTag("particleFlowRecHitHCAL:HFEM"),
    recHitCleaners = cms.VPSet(spikeAndDoubleSpikeCleaner_HFEM),
    seedFinder = localMaxSeeds_HF,
    topoClusterBuilder = topoClusterizer_HF,
    pfClusterBuilder = pfClusterizer_HF
    )

