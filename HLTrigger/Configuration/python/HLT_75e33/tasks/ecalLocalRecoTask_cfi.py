import FWCore.ParameterSet.Config as cms

from ..tasks.ecalRecHitTask_cfi import *
from ..tasks.ecalUncalibRecHitTask_cfi import *

ecalLocalRecoTask = cms.Task(
    ecalRecHitTask,
    ecalUncalibRecHitTask
)
# foo bar baz
# j49SYwXQnKBto
# sqXWOK9BoE8kX
