import FWCore.ParameterSet.Config as cms

from ..tasks.HLTPFClusteringForEgammaL1SeededTask_cfi import *

HLTPFClusteringForEgammaL1Seeded = cms.Sequence(
    HLTPFClusteringForEgammaL1SeededTask
)
