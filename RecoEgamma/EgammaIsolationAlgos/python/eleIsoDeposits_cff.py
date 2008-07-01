import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.egammaIsoSetup_cff import *

from RecoEgamma.EgammaIsolationAlgos.eleIsoDepositTk_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoDepositEcalFromHits_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoDepositHcalFromHits_cff import *

eleIsoDeposits = cms.Sequence(eleIsoDepositTk+eleIsoDepositEcalFromHits+eleIsoDepositHcalFromHits)

