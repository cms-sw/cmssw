import FWCore.ParameterSet.Config as cms

from ..modules.bunchSpacingProducer_cfi import *
from ..sequences.calolocalrecoSequence_cfi import *

localrecoSequence = cms.Sequence(bunchSpacingProducer+calolocalrecoSequence)
