import FWCore.ParameterSet.Config as cms

from ..modules.hltPreEle26WP70L1Seeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle26WP70L1SeededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele26_WP70_L1Seeded = cms.Path(HLTBeginSequence+hltPreEle26WP70L1Seeded+HLTEle26WP70L1SeededSequence+HLTEndSequence)
