import FWCore.ParameterSet.Config as cms

from ..modules.l1tTkIsoEmSingle36Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkIsoEm36Sequence = cms.Sequence(HLTL1Sequence+l1tTkIsoEmSingle36Filter)
