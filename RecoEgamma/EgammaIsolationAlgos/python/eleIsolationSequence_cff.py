import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.eleIsoDeposits_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsoFromDeposits_cff import *

#Standard reco sequence with electrons
eleIsolationTask = cms.Task( 
    eleIsoDepositsTask , 
    eleIsoFromDepositsTask
)
eleIsolationSequence = cms.Sequence(eleIsolationTask) 

