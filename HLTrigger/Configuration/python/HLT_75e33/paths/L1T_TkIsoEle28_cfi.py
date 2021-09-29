import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.L1TTkIsoEle28Sequence_cfi import *

L1T_TkIsoEle28 = cms.Path(
    HLTBeginSequence +
    L1TTkIsoEle28Sequence +
    HLTEndSequence
)
