import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkIsoEm36Sequence_cfi import *

L1T_TkIsoEm36 = cms.Path(HLTBeginSequence+L1TTkIsoEm36Sequence+HLTEndSequence)
