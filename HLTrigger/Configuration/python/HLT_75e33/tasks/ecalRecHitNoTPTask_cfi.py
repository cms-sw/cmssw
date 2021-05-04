import FWCore.ParameterSet.Config as cms

from ..modules.ecalPreshowerRecHit_cfi import *
from ..modules.ecalRecHit_cfi import *

ecalRecHitNoTPTask = cms.Task(
    ecalPreshowerRecHit,
    ecalRecHit
)
