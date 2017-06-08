import FWCore.ParameterSet.Config as cms
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *

particleFlowOOTRecHitECAL = particleFlowRecHitECAL.clone()
particleFlowOOTRecHitECAL.producers[0].qualityTests[1].timingCleaning = cms.bool(False)
particleFlowOOTRecHitECAL.producers[1].qualityTests[1].timingCleaning = cms.bool(False)
