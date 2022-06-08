import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff import *
from RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff import *


import RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff
interestingGedEleIsoDetIdEB = RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone(
    recHitsLabel  = 'ecalRecHit:EcalRecHitsEB',
    emObjectLabel = 'gedGsfElectrons',
    etCandCut     = 0.0,
    energyCut     = 0.095,
    etCut         = 0.0,
    outerRadius   = 0.6,
    innerRadius   = 0.0
)

import RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff
interestingGedEleIsoDetIdEE = RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone(
    recHitsLabel  = 'ecalRecHit:EcalRecHitsEE',
    emObjectLabel = 'gedGsfElectrons',
    etCandCut     = 0.0,
    energyCut     = 0.0,
    etCut         = 0.110,
    outerRadius   = 0.6,
    innerRadius   = 0.0
)

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGedGamIsoDetIdEB = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone(
    recHitsLabel  = 'ecalRecHit:EcalRecHitsEB',
    emObjectLabel = 'gedPhotons',
    etCandCut     = 0.0,
    energyCut     = 0.095,
    etCut         = 0.0,
    outerRadius   = 0.6,
    innerRadius   = 0.0
)

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGedGamIsoDetIdEE = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone(
    recHitsLabel  = 'ecalRecHit:EcalRecHitsEE',
    emObjectLabel = 'gedPhotons',
    etCandCut     = 0.0,
    energyCut     = 0.0,
    etCut         = 0.110,
    outerRadius   = 0.6,
    innerRadius   = 0.0
)
## OOT photons 
interestingOotGamIsoDetIdEB = interestingGedGamIsoDetIdEB.clone(emObjectLabel = 'ootPhotons')
interestingOotGamIsoDetIdEE = interestingGedGamIsoDetIdEE.clone(emObjectLabel = 'ootPhotons')

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGamIsoDetIdEB = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone(
    recHitsLabel  = 'ecalRecHit:EcalRecHitsEB',
    emObjectLabel = 'photons',
    etCandCut     = 0.0,
    energyCut     = 0.095,
    etCut         = 0.0,
    outerRadius   = 0.6,
    innerRadius   = 0.0
)

import RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff
interestingGamIsoDetIdEE = RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone(
    recHitsLabel  = 'ecalRecHit:EcalRecHitsEE',
    emObjectLabel = 'photons',
    etCandCut     = 0.0,
    energyCut     = 0.0,
    etCut         = 0.110,
    outerRadius   = 0.6,
    innerRadius   = 0.0
)

import RecoEgamma.EgammaIsolationAlgos.interestingGedEgammaIsoHCALDetId_cfi
interestingEgammaIsoHCALSel = cms.PSet(
    maxDIEta           = cms.int32(5),
    maxDIPhi           = cms.int32(5),
    minEnergyHB        = cms.double(0.8),
    minEnergyHEDepth1  = cms.double(0.1),
    minEnergyHEDefault = cms.double(0.2),
)
interestingGedEgammaIsoHCALDetId = RecoEgamma.EgammaIsolationAlgos.interestingGedEgammaIsoHCALDetId_cfi.interestingGedEgammaIsoHCALDetId.clone(
    recHitsLabel       = "hbhereco",
    elesLabel          = "gedGsfElectrons",
    phosLabel          = "gedPhotons",
    superClustersLabel = "particleFlowEGamma",
    minSCEt            = 20,
    minEleEt           = 20,
    minPhoEt           = 20,
    hitSelection       = interestingEgammaIsoHCALSel
)

## OOT Photons
interestingOotEgammaIsoHCALDetId = interestingGedEgammaIsoHCALDetId.clone(
    phosLabel          = "ootPhotons",
    elesLabel          = "",
    superClustersLabel = ""
)

import RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoESDetIdModule_cff
interestingOotEgammaIsoESDetId = RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoESDetIdModule_cff.interestingEgammaIsoESDetId.clone(
    eeClusToESMapLabel = "particleFlowClusterOOTECAL",
    ecalPFClustersLabel= "particleFlowClusterOOTECAL",
    phosLabel          = "ootPhotons",
    elesLabel          = "",
    superClustersLabel = "",
    minSCEt            = 500,
    minEleEt           = 20,
    minPhoEt           = 20,
    maxDR              = 0.4
)

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
    interestingOotEgammaIsoESDetId
)
interestingEgammaIsoDetIds = cms.Sequence(interestingEgammaIsoDetIdsTask)

_pp_on_AA_interestingEgammaIsoDetIdsTask = interestingEgammaIsoDetIdsTask.copy()
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotGamIsoDetIdEB)
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotGamIsoDetIdEE)
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotEgammaIsoHCALDetId)
_pp_on_AA_interestingEgammaIsoDetIdsTask.remove(interestingOotEgammaIsoESDetId)

from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
pp_on_AA.toReplaceWith(interestingEgammaIsoDetIdsTask, _pp_on_AA_interestingEgammaIsoDetIdsTask)

from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive

egamma_lowPt_exclusive.toModify(interestingGedEgammaIsoHCALDetId, 
		           minSCEt = 1.0, #default 20
		           minEleEt= 1.0, #default 20
			   minPhoEt= 1.0 #default 20
) 

from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
run3_HB.toModify(interestingEgammaIsoHCALSel,
                 minEnergyHB = 0.1)
