import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoubleEle25CaloIdLPMS2L1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoubleEle25_CaloIdL_PMS2_L1Seeded = cms.Path(
    HLTBeginSequence +
    HLTDoubleEle25CaloIdLPMS2L1SeededSequence +
    HLTEndSequence
)
