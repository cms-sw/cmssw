import FWCore.ParameterSet.Config as cms

from RecoJets.Configuration.CaloTowersRec_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowRecHitPS_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterECAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHCAL_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterPS_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHFEM_cfi import *
from RecoParticleFlow.PFClusterProducer.particleFlowClusterHFHAD_cfi import *

pfClusteringECAL = cms.Sequence(particleFlowRecHitECAL*particleFlowClusterECAL)
#pfClusteringHCAL = cms.Sequence(particleFlowRecHitHCAL*particleFlowClusterHCAL)
pfClusteringHCALall = cms.Sequence(particleFlowClusterHCAL+particleFlowClusterHFHAD+particleFlowClusterHFEM)
pfClusteringHCAL = cms.Sequence(particleFlowRecHitHCAL*pfClusteringHCALall)
#pfClusteringHCAL = cms.Sequence(particleFlowRecHitHCAL*particleFlowClusterHCAL*particleFlowClusterHFHAD*particleFlowClusterHFEM)
pfClusteringPS = cms.Sequence(particleFlowRecHitPS*particleFlowClusterPS)

towerMakerPF = RecoJets.JetProducers.CaloTowerSchemeB_cfi.towerMaker.clone()
towerMakerPF.HBThreshold = 0.4
towerMakerPF.HESThreshold = 0.4
towerMakerPF.HEDThreshold = 0.4


particleFlowCluster = cms.Sequence(
    #caloTowersRec*
    towerMakerPF*
    pfClusteringECAL*
    pfClusteringHCAL*
    pfClusteringPS
)

