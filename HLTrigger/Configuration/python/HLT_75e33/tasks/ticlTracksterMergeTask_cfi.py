import FWCore.ParameterSet.Config as cms

from ..modules.ticlMultiClustersFromTrackstersMerge_cfi import *
from ..modules.ticlTrackstersMerge_cfi import *

ticlTracksterMergeTask = cms.Task(
    ticlMultiClustersFromTrackstersMerge,
    ticlTrackstersMerge
)
