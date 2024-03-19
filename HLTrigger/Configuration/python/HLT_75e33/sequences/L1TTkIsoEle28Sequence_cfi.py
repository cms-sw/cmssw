import FWCore.ParameterSet.Config as cms

from ..modules.l1tTkIsoEleSingle28Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkIsoEle28Sequence = cms.Sequence(HLTL1Sequence+l1tTkIsoEleSingle28Filter)
