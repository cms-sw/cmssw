import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoLocalPixelSequence_cfi import *
from ..sequences.HLTDoLocalStripSequence_cfi import *

HLTItLocalRecoSequence = cms.Sequence(HLTDoLocalPixelSequence+HLTDoLocalStripSequence)
