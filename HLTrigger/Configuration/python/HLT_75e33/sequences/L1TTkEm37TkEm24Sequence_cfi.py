import FWCore.ParameterSet.Config as cms

from ..modules.L1TkEmDouble24Filter_cfi import *
from ..modules.L1TkEmSingle37Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkEm37TkEm24Sequence = cms.Sequence(
    HLTL1Sequence +
    l1tTkEmSingle37Filter +
    l1tTkEmDouble24Filter
)
