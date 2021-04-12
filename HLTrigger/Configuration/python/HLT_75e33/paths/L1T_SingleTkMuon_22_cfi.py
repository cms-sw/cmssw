import FWCore.ParameterSet.Config as cms

from ..modules.hltL1TkSingleMuFiltered22_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

L1T_SingleTkMuon_22 = cms.Path(HLTBeginSequence+hltL1TkSingleMuFiltered22+HLTEndSequence)
