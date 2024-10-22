import FWCore.ParameterSet.Config as cms

from ..modules.l1tTkEleSingle36Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkEle36Sequence = cms.Sequence(HLTL1Sequence+l1tTkEleSingle36Filter)
