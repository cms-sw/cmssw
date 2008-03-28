import FWCore.ParameterSet.Config as cms

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import *
pfClusteringECAL = cms.Sequence(particleFlowRecHitECAL*particleFlowClusterECAL)
pfClusteringHCAL = cms.Sequence(particleFlowRecHitHCAL*particleFlowClusterHCAL)
pfClusteringPS = cms.Sequence(particleFlowRecHitPS*particleFlowClusterPS)
particleFlowCluster = cms.Sequence(pfClusteringECAL*pfClusteringHCAL*pfClusteringPS)

