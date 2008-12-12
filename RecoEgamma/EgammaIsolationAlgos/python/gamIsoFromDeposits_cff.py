import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import *

gamIsoFromDeposits = cms.Sequence( 
    gamIsoFromDepsTk *
    gamIsoFromDepsEcalFromHits * 
    gamIsoFromDepsHcalFromTowers *
    gamIsoFromDepsHcalDepth1FromTowers *
    gamIsoFromDepsHcalDepth2FromTowers
)

