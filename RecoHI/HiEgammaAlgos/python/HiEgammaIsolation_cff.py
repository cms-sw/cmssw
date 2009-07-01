import FWCore.ParameterSet.Config as cms

from RecoHI.HiEgammaAlgos.HiCaloIsolation_cff import *
from RecoHI.HiEgammaAlgos.HiTrackerIsolation_cff import *

hiEgammaIsolationSequenceAll = cms.Sequence(hiCaloIsolationAll+hiTrackerIsolation)
hiEgammaIsolationSequence = cms.Sequence(hiCaloIsolationBckSubtracted+hiTrackerIsolation)


