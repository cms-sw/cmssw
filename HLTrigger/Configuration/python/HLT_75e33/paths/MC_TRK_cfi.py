import FWCore.ParameterSet.Config as cms

from ..sequences.HLTBeginSequence_cfi import *
from ..sequences.HLTTrackingSequence_cfi import *

MC_TRK = cms.Path(
    HLTBeginSequence
    + HLTTrackingSequence
)
