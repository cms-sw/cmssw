import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.egammaIsoSetup_cff import *

from RecoEgamma.EgammaAlgos.gamIsoDepositTk_cff import *
from RecoEgamma.EgammaAlgos.gamIsoDepositEcalFromHits_cff import *
from RecoEgamma.EgammaAlgos.gamIsoDepositHcalFromTowers_cff import *

gamIsoDeposits = cms.Sequence(
    gamIsoDepositTk + 
    gamIsoDepositEcalFromHits + 
    gamIsoDepositHcalFromTowers +
    gamIsoDepositHcalDepth1FromTowers +
    gamIsoDepositHcalDepth2FromTowers
)

