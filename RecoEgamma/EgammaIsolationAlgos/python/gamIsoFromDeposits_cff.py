import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import *

gamIsoFromDeposits = cms.Sequence(gamIsoFromDepsTk*gamIsoFromDepsEcalFromHits*gamIsoFromDepsHcalFromHits)

