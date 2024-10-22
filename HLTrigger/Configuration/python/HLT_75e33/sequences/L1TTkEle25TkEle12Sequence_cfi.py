import FWCore.ParameterSet.Config as cms

from ..modules.l1tTkEleDouble12Filter_cfi import *
from ..modules.l1tTkEleSingle25Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkEle25TkEle12Sequence = cms.Sequence(HLTL1Sequence+l1tTkEleSingle25Filter+l1tTkEleDouble12Filter)
