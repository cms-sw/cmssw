import FWCore.ParameterSet.Config as cms

from ..modules.hltPreDoubleEle2312IsoL1Seeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDoubleEle2312IsoL1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_DoubleEle23_12_Iso_L1Seeded = cms.Path(HLTBeginSequence+hltPreDoubleEle2312IsoL1Seeded+HLTDoubleEle2312IsoL1SeededSequence+HLTEndSequence)
