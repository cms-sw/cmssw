import FWCore.ParameterSet.Config as cms

from ..tasks.HLTDoFullUnpackingEgammaEcalTask_cfi import *

HLTDoFullUnpackingEgammaEcalSequence = cms.Sequence(HLTDoFullUnpackingEgammaEcalTask)
