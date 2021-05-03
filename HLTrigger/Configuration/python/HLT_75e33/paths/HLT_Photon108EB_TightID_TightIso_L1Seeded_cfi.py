import FWCore.ParameterSet.Config as cms

from ..modules.hltPrePhoton108EBTightIDTightIsoL1Seeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton108EBTightIDTightIsoL1SeededSequence_cfi import *

HLT_Photon108EB_TightID_TightIso_L1Seeded = cms.Path(HLTBeginSequence+hltPrePhoton108EBTightIDTightIsoL1Seeded+HLTPhoton108EBTightIDTightIsoL1SeededSequence+HLTEndSequence)
