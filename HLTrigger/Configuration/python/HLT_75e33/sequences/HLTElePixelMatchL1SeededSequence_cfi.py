import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoLocalPixelSequence_cfi import *
from ..sequences.HLTDoLocalStripSequence_cfi import *
from ..tasks.HLTElePixelMatchL1SeededTask_cfi import *

HLTElePixelMatchL1SeededSequence = cms.Sequence(
    HLTDoLocalPixelSequence +
    HLTDoLocalStripSequence,
    HLTElePixelMatchL1SeededTask
)
