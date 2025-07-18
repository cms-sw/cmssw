import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle26WP70UnseededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele26_WP70_Unseeded = cms.Path(
    HLTBeginSequence +
    HLTEle26WP70UnseededSequence +
    HLTEndSequence
)
