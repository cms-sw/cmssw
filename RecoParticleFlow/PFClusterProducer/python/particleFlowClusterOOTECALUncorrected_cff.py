import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALUncorrected_cfi import *

particleFlowClusterOOTECALUncorrected = particleFlowClusterECALUncorrected.clone(
    recHitsSource = "particleFlowRecHitOOTECAL"
)
# foo bar baz
# UPndV7ZgGWvL5
