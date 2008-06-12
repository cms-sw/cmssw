import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import *

eleIsoDeposits = cms.Sequence(*eleIsoDepositTk*eleIsoDepositEcalFromHits*eleIsoDepositHcalFromHits)
