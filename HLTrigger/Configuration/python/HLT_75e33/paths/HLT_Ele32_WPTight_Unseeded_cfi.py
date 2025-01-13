import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle32WPTightUnseededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele32_WPTight_Unseeded = cms.Path(
    HLTBeginSequence +
    HLTEle32WPTightUnseededSequence +
    HLTEndSequence
)
