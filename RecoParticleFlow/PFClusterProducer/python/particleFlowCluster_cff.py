import FWCore.ParameterSet.Config as cms


from RecoParticleFlow.PFClusterProducer.towerMakerPF_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALUncorrected_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHFEM_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHFHAD_cfi import *

pfClusteringECAL = cms.Sequence(particleFlowRecHitECAL*
                                particleFlowClusterECALUncorrected *
                                particleFlowClusterECAL)
pfClusteringPS = cms.Sequence(particleFlowRecHitPS*particleFlowClusterPS)


pfClusteringHBHEHF = cms.Sequence(towerMakerPF*particleFlowRecHitHCAL*particleFlowClusterHCAL+particleFlowClusterHFHAD+particleFlowClusterHFEM)
pfClusteringHO = cms.Sequence(particleFlowRecHitHO*particleFlowClusterHO)


particleFlowClusterWithoutHO = cms.Sequence(
    pfClusteringPS*
    pfClusteringECAL*
    pfClusteringHBHEHF
)

particleFlowCluster = cms.Sequence(
    pfClusteringPS*
    pfClusteringECAL*
    pfClusteringHBHEHF*
    pfClusteringHO 
)


