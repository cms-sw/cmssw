import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle115NonIsoL1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele115_NonIso_L1Seeded = cms.Path(
    HLTBeginSequence +
    HLTEle115NonIsoL1SeededSequence +
    HLTEndSequence
)
