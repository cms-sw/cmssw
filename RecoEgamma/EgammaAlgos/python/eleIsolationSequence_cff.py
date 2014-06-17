import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.eleIsoDeposits_cff import *
from RecoEgamma.EgammaAlgos.eleIsoFromDeposits_cff import *

#Standard reco sequence with electrons
eleIsolationSequence = cms.Sequence( 
    eleIsoDeposits * 
    eleIsoFromDeposits
)

