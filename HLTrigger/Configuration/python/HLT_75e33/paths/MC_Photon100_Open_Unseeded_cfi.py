import FWCore.ParameterSet.Config as cms

from ..modules.hltPrePhoton100OpenUnseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton100OpenUnseededSequence_cfi import *

MC_Photon100_Open_Unseeded = cms.Path(HLTBeginSequence+hltPrePhoton100OpenUnseeded+HLTPhoton100OpenUnseededSequence+HLTEndSequence)
