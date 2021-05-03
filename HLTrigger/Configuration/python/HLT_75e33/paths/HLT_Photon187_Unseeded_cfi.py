import FWCore.ParameterSet.Config as cms

from ..modules.hltPrePhoton187Unseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton187UnseededSequence_cfi import *

HLT_Photon187_Unseeded = cms.Path(HLTBeginSequence+hltPrePhoton187Unseeded+HLTPhoton187UnseededSequence+HLTEndSequence)
