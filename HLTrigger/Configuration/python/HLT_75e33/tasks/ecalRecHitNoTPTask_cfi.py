import FWCore.ParameterSet.Config as cms

from ..modules.hltEcalPreshowerRecHit_cfi import *
from ..modules.ecalRecHit_cfi import *

ecalRecHitNoTPTask = cms.Task(
    hltEcalPreshowerRecHit,
    ecalRecHit
)
