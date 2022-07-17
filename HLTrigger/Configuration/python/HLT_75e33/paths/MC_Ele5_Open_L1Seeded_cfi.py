import FWCore.ParameterSet.Config as cms

from ..modules.hltPreEle5OpenL1Seeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle5OpenL1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

MC_Ele5_Open_L1Seeded = cms.Path(
    HLTBeginSequence +
    hltPreEle5OpenL1Seeded +
    HLTEle5OpenL1SeededSequence +
    HLTEndSequence
)
