import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.egammaIsoSetup_cff import *

from RecoEgamma.EgammaAlgos.eleIsoDepositTk_cff import *
from RecoEgamma.EgammaAlgos.eleIsoDepositEcalFromHits_cff import *
from RecoEgamma.EgammaAlgos.eleIsoDepositHcalFromTowers_cff import *

eleIsoDeposits = cms.Sequence(
    eleIsoDepositTk +
    eleIsoDepositEcalFromHits + 
    eleIsoDepositHcalFromTowers +
    eleIsoDepositHcalDepth1FromTowers +
    eleIsoDepositHcalDepth2FromTowers
)

