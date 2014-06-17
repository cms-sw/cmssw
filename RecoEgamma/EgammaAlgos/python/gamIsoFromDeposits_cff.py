import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.gamIsoFromDepsModules_cff import *

gamIsoFromDeposits = cms.Sequence( 
    gamIsoFromDepsTk *
    gamIsoFromDepsEcalFromHitsByCrystal * 
    gamIsoFromDepsHcalFromTowers *
    gamIsoFromDepsHcalDepth1FromTowers *
    gamIsoFromDepsHcalDepth2FromTowers
)

