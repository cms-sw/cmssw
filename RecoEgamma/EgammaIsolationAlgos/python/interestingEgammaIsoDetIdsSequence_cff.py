import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff import *
from RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff import *


import RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff
interestingGedEleIsoDetIdEB = RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone()
interestingGedEleIsoDetIdEB.recHitsLabel = 'ecalRecHit:EcalRecHitsEB'
interestingGedEleIsoDetIdEB.emObjectLabel = 'gedGsfElectrons'
interestingGedEleIsoDetIdEB.etCandCut = 0.0
interestingGedEleIsoDetIdEB.energyCut = 0.095
interestingGedEleIsoDetIdEB.etCut = 0.0
interestingGedEleIsoDetIdEB.outerRadius = 0.6
interestingGedEleIsoDetIdEB.innerRadius = 0.0

import RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff
interestingGedEleIsoDetIdEE = RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone()
interestingGedEleIsoDetIdEE.recHitsLabel = 'ecalRecHit:EcalRecHitsEE'
interestingGedEleIsoDetIdEE.emObjectLabel = 'gedGsfElectrons'
interestingGedEleIsoDetIdEE.etCandCut = 0.0
interestingGedEleIsoDetIdEE.energyCut = 0.0
interestingGedEleIsoDetIdEE.etCut = 0.110
interestingGedEleIsoDetIdEE.outerRadius = 0.6
interestingGedEleIsoDetIdEE.innerRadius = 0.0

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGedGamIsoDetIdEB = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGedGamIsoDetIdEB.recHitsLabel = 'ecalRecHit:EcalRecHitsEB'
interestingGedGamIsoDetIdEB.emObjectLabel = 'gedPhotons'
interestingGedGamIsoDetIdEB.etCandCut = 0.0
interestingGedGamIsoDetIdEB.energyCut = 0.095
interestingGedGamIsoDetIdEB.etCut = 0.0
interestingGedGamIsoDetIdEB.outerRadius = 0.6
interestingGedGamIsoDetIdEB.innerRadius = 0.0

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGedGamIsoDetIdEE = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGedGamIsoDetIdEE.recHitsLabel = 'ecalRecHit:EcalRecHitsEE'
interestingGedGamIsoDetIdEE.emObjectLabel = 'gedPhotons'
interestingGedGamIsoDetIdEE.etCandCut = 0.0
interestingGedGamIsoDetIdEE.energyCut = 0.0
interestingGedGamIsoDetIdEE.etCut = 0.110
interestingGedGamIsoDetIdEE.outerRadius = 0.6
interestingGedGamIsoDetIdEE.innerRadius = 0.0

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGamIsoDetIdEB = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGamIsoDetIdEB.recHitsLabel = 'ecalRecHit:EcalRecHitsEB'
interestingGamIsoDetIdEB.emObjectLabel = 'photons'
interestingGamIsoDetIdEB.etCandCut = 0.0
interestingGamIsoDetIdEB.energyCut = 0.095
interestingGamIsoDetIdEB.etCut = 0.0
interestingGamIsoDetIdEB.outerRadius = 0.6
interestingGamIsoDetIdEB.innerRadius = 0.0

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGamIsoDetIdEE = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGamIsoDetIdEE.recHitsLabel = 'ecalRecHit:EcalRecHitsEE'
interestingGamIsoDetIdEE.emObjectLabel = 'photons'
interestingGamIsoDetIdEE.etCandCut = 0.0
interestingGamIsoDetIdEE.energyCut = 0.0
interestingGamIsoDetIdEE.etCut = 0.110
interestingGamIsoDetIdEE.outerRadius = 0.6
interestingGamIsoDetIdEE.innerRadius = 0.0

import RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoHCALDetIdModule_cff
interestingGedEgammaIsoHCALDetId = RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoHCALDetIdModule_cff.interestingEgammaIsoHCALDetId.clone()
interestingGedEgammaIsoHCALDetId.recHitsLabel=cms.InputTag("hbhereco")
interestingGedEgammaIsoHCALDetId.elesLabel=cms.InputTag("gedGsfElectrons")
interestingGedEgammaIsoHCALDetId.phosLabel=cms.InputTag("gedPhotons")
interestingGedEgammaIsoHCALDetId.superClustersLabel=cms.InputTag("particleFlowEGamma")
interestingGedEgammaIsoHCALDetId.minSCEt=cms.double(20)
interestingGedEgammaIsoHCALDetId.minEleEt=cms.double(20)
interestingGedEgammaIsoHCALDetId.minPhoEt=cms.double(20)
interestingGedEgammaIsoHCALDetId.maxDIEta=cms.int32(5)
interestingGedEgammaIsoHCALDetId.maxDIPhi=cms.int32(5)
interestingGedEgammaIsoHCALDetId.minEnergyHCAL = cms.double(0.8)


import RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoESDetIdModule_cff
interestingGedEgammaIsoESDetId = RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoESDetIdModule_cff.interestingEgammaIsoESDetId.clone()
interestingGedEgammaIsoESDetId.eeClusToESMapLabel=cms.InputTag("particleFlowClusterECALRemade")
interestingGedEgammaIsoESDetId.ecalPFClustersLabel=cms.InputTag("particleFlowClusterECALRemade")
interestingGedEgammaIsoESDetId.elesLabel=cms.InputTag("gedGsfElectrons")
interestingGedEgammaIsoESDetId.phosLabel=cms.InputTag("gedPhotons")
interestingGedEgammaIsoESDetId.superClustersLabel=cms.InputTag("particleFlowEGamma")
interestingGedEgammaIsoESDetId.minSCEt=cms.double(500)
interestingGedEgammaIsoESDetId.minEleEt=cms.double(20)
interestingGedEgammaIsoESDetId.minPhoEt=cms.double(20)
interestingGedEgammaIsoESDetId.maxDR=cms.double(0.4)

interestingEgammaIsoDetIds = cms.Sequence(
    interestingGedEleIsoDetIdEB *
    interestingGedEleIsoDetIdEE * 
    interestingGedGamIsoDetIdEB * 
    interestingGedGamIsoDetIdEE *   
    interestingGamIsoDetIdEB * 
    interestingGamIsoDetIdEE *
    interestingGedEgammaIsoHCALDetId*
    interestingGedEgammaIsoESDetId
)
