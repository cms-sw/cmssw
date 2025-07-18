import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton108EBTightIDTightIsoUnseededSequence_cfi import *

HLT_Photon108EB_TightID_TightIso_Unseeded = cms.Path(
    HLTBeginSequence +
    HLTPhoton108EBTightIDTightIsoUnseededSequence +
    HLTEndSequence
)
