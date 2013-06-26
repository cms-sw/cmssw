import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff import *
from RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff import *


import RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff
interestingEleIsoDetIdEB = RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone()
interestingEleIsoDetIdEB.recHitsLabel = 'ecalRecHit:EcalRecHitsEB'
interestingEleIsoDetIdEB.emObjectLabel = 'gsfElectrons'
interestingEleIsoDetIdEB.etCandCut = 0.0
interestingEleIsoDetIdEB.energyCut = 0.095
interestingEleIsoDetIdEB.etCut = 0.0
interestingEleIsoDetIdEB.outerRadius = 0.6
interestingEleIsoDetIdEB.innerRadius = 0.0

import RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff
interestingEleIsoDetIdEE = RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff.interestingEleIsoDetId.clone()
interestingEleIsoDetIdEE.recHitsLabel = 'ecalRecHit:EcalRecHitsEE'
interestingEleIsoDetIdEE.emObjectLabel = 'gsfElectrons'
interestingEleIsoDetIdEE.etCandCut = 0.0
interestingEleIsoDetIdEE.energyCut = 0.0
interestingEleIsoDetIdEE.etCut = 0.110
interestingEleIsoDetIdEE.outerRadius = 0.6
interestingEleIsoDetIdEE.innerRadius = 0.0

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

interestingEgammaIsoDetIds = cms.Sequence(
    interestingEleIsoDetIdEB *
    interestingEleIsoDetIdEE * 
    interestingGamIsoDetIdEB * 
    interestingGamIsoDetIdEE
)
