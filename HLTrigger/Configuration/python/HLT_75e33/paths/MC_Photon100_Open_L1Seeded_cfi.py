import FWCore.ParameterSet.Config as cms

from ..modules.hltPrePhoton100OpenL1Seeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton100OpenL1SeededSequence_cfi import *

MC_Photon100_Open_L1Seeded = cms.Path(HLTBeginSequence+hltPrePhoton100OpenL1Seeded+HLTPhoton100OpenL1SeededSequence+HLTEndSequence)
