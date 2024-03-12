import FWCore.ParameterSet.Config as cms

from ..modules.ecalDetailedTimeRecHit_cfi import *
from ..tasks.ecalRecHitNoTPTask_cfi import *

ecalRecHitTask = cms.Task(
    ecalDetailedTimeRecHit,
    ecalRecHitNoTPTask
)
# foo bar baz
# DmYG3FRquzwve
