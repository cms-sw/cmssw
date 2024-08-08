import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *

HLTCalolocalrecoSequence = cms.Sequence(HLTDoFullUnpackingEgammaEcalSequence+HLTDoLocalHcalSequence)
