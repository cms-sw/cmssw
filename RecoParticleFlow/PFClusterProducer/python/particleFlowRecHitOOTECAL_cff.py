import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *

particleFlowRecHitOOTECAL = particleFlowRecHitECAL.clone( 
    producers = {0 : dict(qualityTests = {1 : dict(timingCleaning = False) } ), 
		 1 : dict(qualityTests = {1 : dict(timingCleaning = False) } )}
)
