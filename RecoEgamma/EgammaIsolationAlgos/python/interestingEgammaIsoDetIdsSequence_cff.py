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

## OOT photons 
interestingOotGamIsoDetIdEB = interestingGedGamIsoDetIdEB.clone(emObjectLabel = 'ootPhotons')
interestingOotGamIsoDetIdEE = interestingGedGamIsoDetIdEE.clone(emObjectLabel = 'ootPhotons')

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

import RecoEgamma.EgammaIsolationAlgos.interestingGedEgammaIsoHCALDetId_cfi
interestingGedEgammaIsoHCALDetId = RecoEgamma.EgammaIsolationAlgos.interestingGedEgammaIsoHCALDetId_cfi.interestingGedEgammaIsoHCALDetId.clone()
interestingEgammaIsoHCALSel = cms.PSet(
  maxDIEta=cms.int32(5),
  maxDIPhi=cms.int32(5),
  minEnergyHB = cms.double(0.8),
  minEnergyHEDepth1 = cms.double(0.1),
  minEnergyHEDefault = cms.double(0.2),
)
interestingGedEgammaIsoHCALDetId.recHitsLabel=cms.InputTag("hbhereco")
interestingGedEgammaIsoHCALDetId.elesLabel=cms.InputTag("gedGsfElectrons")
interestingGedEgammaIsoHCALDetId.phosLabel=cms.InputTag("gedPhotons")
interestingGedEgammaIsoHCALDetId.superClustersLabel=cms.InputTag("particleFlowEGamma")
interestingGedEgammaIsoHCALDetId.minSCEt=cms.double(20)
interestingGedEgammaIsoHCALDetId.minEleEt=cms.double(20)
interestingGedEgammaIsoHCALDetId.minPhoEt=cms.double(20)
interestingGedEgammaIsoHCALDetId.hitSelection=interestingEgammaIsoHCALSel

## OOT Photons
interestingOotEgammaIsoHCALDetId = interestingGedEgammaIsoHCALDetId.clone()
interestingOotEgammaIsoHCALDetId.phosLabel=cms.InputTag("ootPhotons")
interestingOotEgammaIsoHCALDetId.elesLabel=cms.InputTag("")
interestingOotEgammaIsoHCALDetId.superClustersLabel=cms.InputTag("")

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

## OOT Photons
interestingOotEgammaIsoESDetId = interestingGedEgammaIsoESDetId.clone()
interestingOotEgammaIsoESDetId.eeClusToESMapLabel=cms.InputTag("particleFlowClusterOOTECAL")
interestingOotEgammaIsoESDetId.ecalPFClustersLabel=cms.InputTag("particleFlowClusterOOTECAL")
interestingOotEgammaIsoESDetId.phosLabel=cms.InputTag("ootPhotons")
interestingOotEgammaIsoESDetId.elesLabel=cms.InputTag("")
interestingOotEgammaIsoESDetId.superClustersLabel=cms.InputTag("")

interestingEgammaIsoDetIdsTask = cms.Task(
    interestingGedEleIsoDetIdEB ,
    interestingGedEleIsoDetIdEE , 
    interestingGedGamIsoDetIdEB , 
    interestingGedGamIsoDetIdEE ,   
    interestingOotGamIsoDetIdEB , 
    interestingOotGamIsoDetIdEE ,   
    interestingGamIsoDetIdEB , 
    interestingGamIsoDetIdEE ,
    interestingGedEgammaIsoHCALDetId,
    interestingOotEgammaIsoHCALDetId,
    interestingGedEgammaIsoESDetId,
    interestingOotEgammaIsoESDetId
)
interestingEgammaIsoDetIds = cms.Sequence(interestingEgammaIsoDetIdsTask)

_pp_on_AA_interestingEgammaIsoDetIdsTask = interestingEgammaIsoDetIdsTask.copy()
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotGamIsoDetIdEB)
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotGamIsoDetIdEE)
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotEgammaIsoHCALDetId)
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotEgammaIsoESDetId)

from Configuration.Eras.Modifier_pp_on_AA_2018_cff import pp_on_AA_2018
pp_on_AA_2018.toReplaceWith(interestingEgammaIsoDetIdsTask, _pp_on_AA_interestingEgammaIsoDetIdsTask)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
egamma_lowPt_exclusive.toModify(interestingGedEgammaIsoESDetId,
			   minSCEt   = 1.0, #default 500
			   minEleEt  = 1.0, #default 20
			   minPhoEt  = 1.0 #default 20
)
egamma_lowPt_exclusive.toModify(interestingGedEgammaIsoHCALDetId, 
		           minSCEt = 1.0, #default 20
		           minEleEt= 1.0, #default 20
			   minPhoEt= 1.0 #default 20
) 

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(interestingEgammaIsoHCALSel,
                 minEnergyHB = 0.1)
