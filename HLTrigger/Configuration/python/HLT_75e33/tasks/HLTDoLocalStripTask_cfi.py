import FWCore.ParameterSet.Config as cms

from ..modules.siPhase2Clusters_cfi import *
from ..modules.siStripDigis_cfi import *

HLTDoLocalStripTask = cms.Task(
    siPhase2Clusters,
    siStripDigis
)
