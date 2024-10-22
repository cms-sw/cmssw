import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaIsoSetup_cff import *

from RecoEgamma.EgammaIsolationAlgos.eleIsoDepositTk_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoDepositEcalFromHits_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoDepositHcalFromTowers_cff import *

eleIsoDepositsTask = cms.Task(
    eleIsoDepositTk ,
    eleIsoDepositEcalFromHits , 
    eleIsoDepositHcalFromTowers ,
    eleIsoDepositHcalDepth1FromTowers ,
    eleIsoDepositHcalDepth2FromTowers
)
