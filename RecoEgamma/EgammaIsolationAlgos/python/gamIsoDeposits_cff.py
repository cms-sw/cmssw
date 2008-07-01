import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaIsoSetup_cff import *

from RecoEgamma.EgammaIsolationAlgos.gamIsoDepositTk_cff import *
from RecoEgamma.EgammaIsolationAlgos.gamIsoDepositEcalFromHits_cff import *
from RecoEgamma.EgammaIsolationAlgos.gamIsoDepositHcalFromHits_cff import *

gamIsoDeposits = cms.Sequence(gamIsoDepositTk+gamIsoDepositEcalFromHits+gamIsoDepositHcalFromHits)

