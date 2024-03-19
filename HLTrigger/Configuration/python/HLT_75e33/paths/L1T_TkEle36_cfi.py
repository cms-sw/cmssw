import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkEle36Sequence_cfi import *

L1T_TkEle36 = cms.Path(HLTBeginSequence+L1TTkEle36Sequence+HLTEndSequence)
