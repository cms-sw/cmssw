import FWCore.ParameterSet.Config as cms

from ..modules.ecalCompactTrigPrim_cfi import *
from ..modules.ecalDetailedTimeRecHit_cfi import *
from ..modules.ecalTPSkim_cfi import *
from ..tasks.ecalRecHitNoTPTask_cfi import *

ecalRecHitTask = cms.Task(ecalCompactTrigPrim, ecalDetailedTimeRecHit, ecalRecHitNoTPTask, ecalTPSkim)
