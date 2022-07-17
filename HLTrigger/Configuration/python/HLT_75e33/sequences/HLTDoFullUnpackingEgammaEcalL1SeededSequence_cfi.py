import FWCore.ParameterSet.Config as cms

from ..tasks.HLTDoFullUnpackingEgammaEcalL1SeededTask_cfi import *

HLTDoFullUnpackingEgammaEcalL1SeededSequence = cms.Sequence(
    HLTDoFullUnpackingEgammaEcalL1SeededTask
)
