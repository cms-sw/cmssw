import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.interestingEleIsoDetIdModule_cff import *
from RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff import *


import RecoEgamma.EgammaAlgos.interestingEleIsoDetIdModule_cff
interestingGedEleIsoDetIdEB = RecoEgamma.EgammaAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone()
interestingGedEleIsoDetIdEB.recHitsLabel = 'ecalRecHit:EcalRecHitsEB'
interestingGedEleIsoDetIdEB.emObjectLabel = 'gedGsfElectrons'
interestingGedEleIsoDetIdEB.etCandCut = 0.0
interestingGedEleIsoDetIdEB.energyCut = 0.095
interestingGedEleIsoDetIdEB.etCut = 0.0
interestingGedEleIsoDetIdEB.outerRadius = 0.6
interestingGedEleIsoDetIdEB.innerRadius = 0.0

import RecoEgamma.EgammaAlgos.interestingEleIsoDetIdModule_cff
interestingGedEleIsoDetIdEE = RecoEgamma.EgammaAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone()
interestingGedEleIsoDetIdEE.recHitsLabel = 'ecalRecHit:EcalRecHitsEE'
interestingGedEleIsoDetIdEE.emObjectLabel = 'gedGsfElectrons'
interestingGedEleIsoDetIdEE.etCandCut = 0.0
interestingGedEleIsoDetIdEE.energyCut = 0.0
interestingGedEleIsoDetIdEE.etCut = 0.110
interestingGedEleIsoDetIdEE.outerRadius = 0.6
interestingGedEleIsoDetIdEE.innerRadius = 0.0

import RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff
interestingGedGamIsoDetIdEB = RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGedGamIsoDetIdEB.recHitsLabel = 'ecalRecHit:EcalRecHitsEB'
interestingGedGamIsoDetIdEB.emObjectLabel = 'gedPhotons'
interestingGedGamIsoDetIdEB.etCandCut = 0.0
interestingGedGamIsoDetIdEB.energyCut = 0.095
interestingGedGamIsoDetIdEB.etCut = 0.0
interestingGedGamIsoDetIdEB.outerRadius = 0.6
interestingGedGamIsoDetIdEB.innerRadius = 0.0

import RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff
interestingGedGamIsoDetIdEE = RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGedGamIsoDetIdEE.recHitsLabel = 'ecalRecHit:EcalRecHitsEE'
interestingGedGamIsoDetIdEE.emObjectLabel = 'gedPhotons'
interestingGedGamIsoDetIdEE.etCandCut = 0.0
interestingGedGamIsoDetIdEE.energyCut = 0.0
interestingGedGamIsoDetIdEE.etCut = 0.110
interestingGedGamIsoDetIdEE.outerRadius = 0.6
interestingGedGamIsoDetIdEE.innerRadius = 0.0

import RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff
interestingGamIsoDetIdEB = RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGamIsoDetIdEB.recHitsLabel = 'ecalRecHit:EcalRecHitsEB'
interestingGamIsoDetIdEB.emObjectLabel = 'photons'
interestingGamIsoDetIdEB.etCandCut = 0.0
interestingGamIsoDetIdEB.energyCut = 0.095
interestingGamIsoDetIdEB.etCut = 0.0
interestingGamIsoDetIdEB.outerRadius = 0.6
interestingGamIsoDetIdEB.innerRadius = 0.0

import RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff
interestingGamIsoDetIdEE = RecoEgamma.EgammaAlgos.interestingGamIsoDetIdModule_cff.interestingGamIsoDetId.clone()
interestingGamIsoDetIdEE.recHitsLabel = 'ecalRecHit:EcalRecHitsEE'
interestingGamIsoDetIdEE.emObjectLabel = 'photons'
interestingGamIsoDetIdEE.etCandCut = 0.0
interestingGamIsoDetIdEE.energyCut = 0.0
interestingGamIsoDetIdEE.etCut = 0.110
interestingGamIsoDetIdEE.outerRadius = 0.6
interestingGamIsoDetIdEE.innerRadius = 0.0

interestingEgammaIsoDetIds = cms.Sequence(
    interestingGedEleIsoDetIdEB *
    interestingGedEleIsoDetIdEE * 
    interestingGedGamIsoDetIdEB * 
    interestingGedGamIsoDetIdEE *   
    interestingGamIsoDetIdEB * 
    interestingGamIsoDetIdEE
)
