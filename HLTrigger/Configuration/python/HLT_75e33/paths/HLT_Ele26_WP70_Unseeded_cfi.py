import FWCore.ParameterSet.Config as cms

from ..modules.hltPreEle26WP70Unseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle26WP70UnseededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Ele26_WP70_Unseeded = cms.Path(HLTBeginSequence+hltPreEle26WP70Unseeded+HLTEle26WP70UnseededSequence+HLTEndSequence)
