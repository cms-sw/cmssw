import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaIsolationAlgos.gamIsolationSequence_cff import *
from RecoEgamma.EgammaIsolationAlgos.eleIsolationSequence_cff import *

#Standard reco sequence with both electrons and photons
egammaIsolationSequencePAT = cms.Sequence(
    eleIsolationSequence *
    gamIsolationSequence
)

