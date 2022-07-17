import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkEm37TkEm24Sequence_cfi import *

L1T_TkEm37TkEm24 = cms.Path(
    HLTBeginSequence +
    L1TTkEm37TkEm24Sequence +
    HLTEndSequence
)
