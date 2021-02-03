import FWCore.ParameterSet.Config as cms

from ..modules.preDuplicateMergingDisplacedTracks_cfi import *
from ..tasks.displacedTracksTask_cfi import *
from ..tasks.muonSeededStepDisplacedTask_cfi import *

iterDisplcedTrackingTask = cms.Task(displacedTracksTask, muonSeededStepDisplacedTask, preDuplicateMergingDisplacedTracks)
