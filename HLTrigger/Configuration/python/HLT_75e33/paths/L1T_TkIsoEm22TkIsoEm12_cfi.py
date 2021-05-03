import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkIsoEm22TkIsoEm12Sequence_cfi import *

L1T_TkIsoEm22TkIsoEm12 = cms.Path(HLTBeginSequence+L1TTkIsoEm22TkIsoEm12Sequence+HLTEndSequence)
