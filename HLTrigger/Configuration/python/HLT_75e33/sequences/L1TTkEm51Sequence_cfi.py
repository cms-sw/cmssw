import FWCore.ParameterSet.Config as cms

from ..modules.l1tTkEmSingle51Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkEm51Sequence = cms.Sequence(HLTL1Sequence+l1tTkEmSingle51Filter)
