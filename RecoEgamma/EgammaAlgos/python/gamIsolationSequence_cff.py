import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.gamIsoDeposits_cff import *
from RecoEgamma.EgammaAlgos.gamIsoFromDeposits_cff import *

#Standard reco sequence with photons
gamIsolationSequence = cms.Sequence(
    gamIsoDeposits * 
    gamIsoFromDeposits
)

