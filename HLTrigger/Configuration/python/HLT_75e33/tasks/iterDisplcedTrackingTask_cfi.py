import FWCore.ParameterSet.Config as cms

from ..modules.preDuplicateMergingDisplacedTracks_cfi import *
from ..tasks.displacedTracksTask_cfi import *

iterDisplcedTrackingTask = cms.Task(displacedTracksTask, preDuplicateMergingDisplacedTracks)
