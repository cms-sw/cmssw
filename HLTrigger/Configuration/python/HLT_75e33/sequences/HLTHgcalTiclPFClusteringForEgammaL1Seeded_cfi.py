import FWCore.ParameterSet.Config as cms

from ..tasks.HLTHgcalTiclPFClusteringForEgammaL1SeededTask_cfi import *

HLTHgcalTiclPFClusteringForEgammaL1Seeded = cms.Sequence(
    HLTHgcalTiclPFClusteringForEgammaL1SeededTask
)
