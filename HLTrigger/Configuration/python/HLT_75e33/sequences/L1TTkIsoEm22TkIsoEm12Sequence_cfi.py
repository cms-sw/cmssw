import FWCore.ParameterSet.Config as cms

from ..modules.l1tTkIsoEmDouble12Filter_cfi import *
from ..modules.l1tTkIsoEmSingle22Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkIsoEm22TkIsoEm12Sequence = cms.Sequence(HLTL1Sequence+l1tTkIsoEmSingle22Filter+l1tTkIsoEmDouble12Filter)
