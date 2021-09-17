import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALUncorrected_cfi import *

particleFlowClusterOOTECALUncorrected = particleFlowClusterECALUncorrected.clone(
    recHitsSource = "particleFlowRecHitOOTECAL"
)
