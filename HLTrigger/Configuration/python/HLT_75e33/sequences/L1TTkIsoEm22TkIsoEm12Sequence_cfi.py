import FWCore.ParameterSet.Config as cms

from ..modules.L1TkIsoEmDouble12Filter_cfi import *
from ..modules.L1TkIsoEmSingle22Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkIsoEm22TkIsoEm12Sequence = cms.Sequence(HLTL1Sequence+L1TkIsoEmSingle22Filter+L1TkIsoEmDouble12Filter)
