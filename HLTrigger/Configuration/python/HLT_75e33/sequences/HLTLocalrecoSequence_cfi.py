import FWCore.ParameterSet.Config as cms

from ..modules.hltBunchSpacingProducer_cfi import *
from ..sequences.HLTCalolocalrecoSequence_cfi import *

HLTLocalrecoSequence = cms.Sequence(bunchSpacingProducer+HLTCalolocalrecoSequence)
