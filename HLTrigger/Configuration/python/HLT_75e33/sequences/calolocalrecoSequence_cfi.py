import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoFullUnpackingEgammaEcalSequence_cfi import *
from ..sequences.HLTDoLocalHcalSequence_cfi import *

calolocalrecoSequence = cms.Sequence(HLTDoFullUnpackingEgammaEcalSequence+HLTDoLocalHcalSequence)
