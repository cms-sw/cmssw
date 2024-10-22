import FWCore.ParameterSet.Config as cms

from ..modules.hltTriggerType_cfi import *
from ..sequences.HLTBeamSpotSequence_cfi import *

HLTBeginSequence = cms.Sequence(hltTriggerType+HLTBeamSpotSequence)
