import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoubleEle2312IsoL1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoubleEle23_12_Iso_L1Seeded = cms.Path(
    HLTBeginSequence +
    HLTDoubleEle2312IsoL1SeededSequence +
    HLTEndSequence
)
