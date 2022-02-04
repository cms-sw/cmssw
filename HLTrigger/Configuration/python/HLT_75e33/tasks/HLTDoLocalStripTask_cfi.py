import FWCore.ParameterSet.Config as cms

from ..modules.siPhase2Clusters_cfi import *

HLTDoLocalStripTask = cms.Task(
    siPhase2Clusters,
)
