import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import *

egammaIsoFromDeposits = cms.Sequence(gamIsoFromDepsTk*gamIsoFromDepsEcalFromHits*gamIsoFromDepsHcalFromHits)

