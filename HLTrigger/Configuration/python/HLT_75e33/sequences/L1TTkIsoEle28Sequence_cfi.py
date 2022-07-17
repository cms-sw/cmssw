import FWCore.ParameterSet.Config as cms

from ..modules.L1TkIsoEleSingle28Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkIsoEle28Sequence = cms.Sequence(
    HLTL1Sequence +
    L1TkIsoEleSingle28Filter
)
