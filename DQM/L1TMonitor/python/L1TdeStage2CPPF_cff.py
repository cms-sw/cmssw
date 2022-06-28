import FWCore.ParameterSet.Config as cms

from DQM.L1TMonitor.L1TdeStage2CPPF_cfi import *


# sequences
l1tdeStage2CppfOnlineDQMSeq = cms.Sequence(
    l1tdeStage2Cppf 
)

