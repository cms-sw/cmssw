import FWCore.ParameterSet.Config as cms

from ..modules.muonSeededSeedsOutInDisplaced_cfi import *
from ..modules.muonSeededTrackCandidatesOutInDisplaced_cfi import *
from ..modules.muonSeededTracksOutInDisplaced_cfi import *
from ..tasks.muonSeededStepCoreInOutTask_cfi import *

muonSeededStepCoreDisplacedTask = cms.Task(muonSeededSeedsOutInDisplaced, muonSeededStepCoreInOutTask, muonSeededTrackCandidatesOutInDisplaced, muonSeededTracksOutInDisplaced)
