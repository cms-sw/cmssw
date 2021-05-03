import FWCore.ParameterSet.Config as cms

from ..modules.L1TkEleSingle36Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkEle36Sequence = cms.Sequence(HLTL1Sequence+L1TkEleSingle36Filter)
