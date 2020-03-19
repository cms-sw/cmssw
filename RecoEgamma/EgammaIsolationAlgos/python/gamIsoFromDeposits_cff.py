import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamIsoFromDepsModules_cff import *

gamIsoFromDepositsTask = cms.Task( 
    gamIsoFromDepsTk ,
    gamIsoFromDepsEcalFromHitsByCrystal , 
    gamIsoFromDepsHcalFromTowers ,
    gamIsoFromDepsHcalDepth1FromTowers ,
    gamIsoFromDepsHcalDepth2FromTowers
)
