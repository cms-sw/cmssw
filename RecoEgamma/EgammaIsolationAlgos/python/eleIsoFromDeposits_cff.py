import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDepsModules_cff import *

eleIsoFromDeposits = cms.Sequence( 
    eleIsoFromDepsTk * 
    eleIsoFromDepsEcalFromHits * 
    eleIsoFromDepsHcalFromTowers *
    eleIsoFromDepsHcalDepth1FromTowers *
    eleIsoFromDepsHcalDepth2FromTowers
)

