import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkEm51Sequence_cfi import *

L1T_TkEm51 = cms.Path(HLTBeginSequence+L1TTkEm51Sequence+HLTEndSequence)
