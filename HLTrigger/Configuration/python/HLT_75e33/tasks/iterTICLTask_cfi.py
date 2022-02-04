import FWCore.ParameterSet.Config as cms

from ..tasks.ticlEMStepTask_cfi import *
from ..tasks.ticlHADStepTask_cfi import *
from ..tasks.ticlLayerTileTask_cfi import *
from ..tasks.ticlPFTask_cfi import *
from ..tasks.ticlTracksterMergeTask_cfi import *
from ..tasks.ticlTrkEMStepTask_cfi import *
from ..tasks.ticlTrkStepTask_cfi import *

iterTICLTask = cms.Task(
    ticlEMStepTask,
    ticlHADStepTask,
    ticlLayerTileTask,
    ticlPFTask,
    ticlTracksterMergeTask,
    ticlTrkEMStepTask,
    ticlTrkStepTask
)
