import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import *

eleIsoFromDeposits = cms.Sequence(eleIsoFromDepsTk*eleIsoFromDepsEcalFromHits*eleIsoFromDepsHcalFromHits)

