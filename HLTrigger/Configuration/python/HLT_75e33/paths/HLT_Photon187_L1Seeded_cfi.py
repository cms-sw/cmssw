import FWCore.ParameterSet.Config as cms

from ..modules.hltPrePhoton187L1Seeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton187L1SeededSequence_cfi import *

HLT_Photon187_L1Seeded = cms.Path(
    HLTBeginSequence +
    hltPrePhoton187L1Seeded +
    HLTPhoton187L1SeededSequence +
    HLTEndSequence
)
