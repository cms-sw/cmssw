import FWCore.ParameterSet.Config as cms

from ..modules.hltL1DoubleNNTau52_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1T_DoubleNNTau52 = cms.Path(HLTL1Sequence+hltL1DoubleNNTau52)
