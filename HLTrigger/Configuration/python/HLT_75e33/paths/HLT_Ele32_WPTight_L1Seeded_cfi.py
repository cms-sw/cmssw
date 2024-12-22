import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle32WPTightL1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele32_WPTight_L1Seeded = cms.Path(
    HLTBeginSequence +
    HLTEle32WPTightL1SeededSequence +
    HLTEndSequence
)
