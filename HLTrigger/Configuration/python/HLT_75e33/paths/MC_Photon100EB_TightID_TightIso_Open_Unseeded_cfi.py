import FWCore.ParameterSet.Config as cms

from ..modules.hltPrePhoton100EBTightIDTightIsoOpenUnseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *
from ..sequences.HLTPhoton100EBTightIDTightIsoOpenUnseededSequence_cfi import *

MC_Photon100EB_TightID_TightIso_Open_Unseeded = cms.Path(
    HLTBeginSequence +
    hltPrePhoton100EBTightIDTightIsoOpenUnseeded +
    HLTPhoton100EBTightIDTightIsoOpenUnseededSequence +
    HLTEndSequence
)
