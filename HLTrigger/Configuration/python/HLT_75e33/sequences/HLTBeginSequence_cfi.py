import FWCore.ParameterSet.Config as cms

from ..modules.hltTriggerType_cfi import *
from ..sequences.HLTBeamSpot_cfi import *
from ..sequences.HLTL1UnpackerSequence_cfi import *

HLTBeginSequence = cms.Sequence(hltTriggerType+HLTL1UnpackerSequence+HLTBeamSpot)
