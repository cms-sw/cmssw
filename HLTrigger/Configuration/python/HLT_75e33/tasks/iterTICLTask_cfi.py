import FWCore.ParameterSet.Config as cms

from ..tasks.ticlLayerTileTask_cfi import *
from ..tasks.ticlPFTask_cfi import *
from ..tasks.ticlTracksterMergeTask_cfi import *
from ..tasks.ticlCLUE3DHighStepTask_cfi import *

iterTICLTask = cms.Task(
    ticlLayerTileTask,
    ticlTrackstersCLUE3DHighStepTask,
    ticlPFTask,
    ticlTracksterMergeTask,
)
