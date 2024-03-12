import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import *

eleIsoFromDepositsTask = cms.Task( 
    eleIsoFromDepsTk , 
    eleIsoFromDepsEcalFromHitsByCrystal , 
    eleIsoFromDepsHcalFromTowers ,
    eleIsoFromDepsHcalDepth1FromTowers ,
    eleIsoFromDepsHcalDepth2FromTowers
)
# foo bar baz
# l2Gza7i0XUC54
