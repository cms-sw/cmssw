import FWCore.ParameterSet.Config as cms

from RecoEgamma.EgammaAlgos.gamIsolationSequence_cff import *
from RecoEgamma.EgammaAlgos.eleIsolationSequence_cff import *

#Standard reco sequence with both electrons and photons
egammaIsolationSequencePAT = cms.Sequence(
    eleIsolationSequence *
    gamIsolationSequence
)

