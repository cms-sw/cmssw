import FWCore.ParameterSet.Config as cms


#from RecoParticleFlow.PFClusterProducer.towerMakerPF_cfi import *
#from RecoParticleFlow.PFClusterProducer.particleFlowCaloResolution_cfi import _timeResolutionHCALMaxSample

from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHBHE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import *

from RecoParticleFlow.PFClusterProducer.particleFlowClusterECALUncorrected_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import *


from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHE_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHBHETimeSelected_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHF_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHO_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import *


pfClusteringECAL = cms.Sequence(particleFlowRecHitECAL*
                                particleFlowClusterECALUncorrected *
                                particleFlowClusterECAL)
pfClusteringPS = cms.Sequence(particleFlowRecHitPS*particleFlowClusterPS)


#pfClusteringHBHEHF = cms.Sequence(towerMakerPF*particleFlowRecHitHCAL*particleFlowClusterHCAL+particleFlowClusterHFHAD+particleFlowClusterHFEM)
pfClusteringHBHEHF = cms.Sequence(particleFlowRecHitHBHE*particleFlowRecHitHF*particleFlowClusterHBHE*particleFlowClusterHF*particleFlowClusterHCAL)
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


