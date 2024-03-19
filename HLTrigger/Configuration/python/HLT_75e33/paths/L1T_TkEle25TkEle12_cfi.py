import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkEle25TkEle12Sequence_cfi import *

L1T_TkEle25TkEle12 = cms.Path(HLTBeginSequence+L1TTkEle25TkEle12Sequence+HLTEndSequence)
