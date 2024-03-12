import FWCore.ParameterSet.Config as cms

from ..modules.hltL1SingleNNTau150_cfi import *
from ..sequences.HLTL1Sequence_cfi import *

L1T_SingleNNTau150 = cms.Path(HLTL1Sequence+hltL1SingleNNTau150)
