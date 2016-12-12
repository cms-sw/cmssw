import FWCore.ParameterSet.Config as cms

from L1Trigger.L1TNtuples.genTree_cfi import *

L1NtupleGEN = cms.Sequence(
    genTree
)
