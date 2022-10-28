import FWCore.ParameterSet.Config as cms

from ..modules.hltPreDiphoton3023IsoCaloIdUnseeded_cfi import *
from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTDiphoton3023IsoCaloIdUnseededSequence_cfi import *
from ..sequences.HLTEndSequence_cfi import *

HLT_Diphoton30_23_IsoCaloId_Unseeded = cms.Path(
    HLTBeginSequence +
    hltPreDiphoton3023IsoCaloIdUnseeded +
    HLTDiphoton3023IsoCaloIdUnseededSequence +
    HLTEndSequence
)
