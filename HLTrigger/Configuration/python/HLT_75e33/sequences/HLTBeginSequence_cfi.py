import FWCore.ParameterSet.Config as cms

from ..modules.hltTriggerType_cfi import *
from ..sequences.HLTBeamSpot_cfi import *

HLTBeginSequence = cms.Sequence(hltTriggerType+HLTBeamSpot)
