import FWCore.ParameterSet.Config as cms

from ..modules.hltPreEle5OpenUnseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEle5OpenUnseededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

MC_Ele5_Open_Unseeded = cms.Path(HLTBeginSequence+hltPreEle5OpenUnseeded+HLTEle5OpenUnseededSequence+HLTEndSequence)
