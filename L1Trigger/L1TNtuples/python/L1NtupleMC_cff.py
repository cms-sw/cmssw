import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.l1GeneratorTree_cfi import *

L1NtupleMC = cms.Sequence(
    l1GeneratorTree
)
