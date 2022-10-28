import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkIsoEle22TkEm12Sequence_cfi import *

L1T_TkIsoEle22TkEm12 = cms.Path(
    HLTBeginSequence +
    L1TTkIsoEle22TkEm12Sequence +
    HLTEndSequence
)
