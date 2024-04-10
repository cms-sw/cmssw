import FWCore.ParameterSet.Config as cms

from ..modules.l1tTkEmDouble24Filter_cfi import *
from ..modules.l1tTkEmSingle37Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkEm37TkEm24Sequence = cms.Sequence(HLTL1Sequence+l1tTkEmSingle37Filter+l1tTkEmDouble24Filter)
