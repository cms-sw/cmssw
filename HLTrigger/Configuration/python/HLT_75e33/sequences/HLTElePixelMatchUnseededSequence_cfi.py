import FWCore.ParameterSet.Config as cms

from ..sequences.HLTDoLocalPixelSequence_cfi import *
from ..sequences.HLTDoLocalStripSequence_cfi import *
from ..tasks.HLTElePixelMatchUnseededTask_cfi import *

HLTElePixelMatchUnseededSequence = cms.Sequence(
    HLTDoLocalPixelSequence +
    HLTDoLocalStripSequence,
    HLTElePixelMatchUnseededTask
)
