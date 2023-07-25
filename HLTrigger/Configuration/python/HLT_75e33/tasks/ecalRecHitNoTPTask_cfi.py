import FWCore.ParameterSet.Config as cms

from ..modules.ecalRecHit_cfi import *

ecalRecHitNoTPTask = cms.Task(
    ecalRecHit
)
