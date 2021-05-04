import FWCore.ParameterSet.Config as cms

from ..modules.ecalDetIdToBeRecovered_cfi import *
from ..tasks.ecalMultiFitUncalibRecHitTask_cfi import *

ecalUncalibRecHitTask = cms.Task(
    ecalDetIdToBeRecovered,
    ecalMultiFitUncalibRecHitTask
)
