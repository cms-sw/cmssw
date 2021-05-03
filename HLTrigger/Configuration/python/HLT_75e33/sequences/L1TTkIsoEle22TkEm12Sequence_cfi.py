import FWCore.ParameterSet.Config as cms

from ..modules.L1TkEmDouble12Filter_cfi import *
from ..modules.L1TkIsoEleSingle22Filter_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1TTkIsoEle22TkEm12Sequence = cms.Sequence(HLTL1Sequence+L1TkIsoEleSingle22Filter+L1TkEmDouble12Filter)
