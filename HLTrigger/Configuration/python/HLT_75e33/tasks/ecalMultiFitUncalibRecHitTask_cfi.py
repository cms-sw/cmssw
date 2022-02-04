import FWCore.ParameterSet.Config as cms

from ..modules.ecalMultiFitUncalibRecHit_cfi import *

ecalMultiFitUncalibRecHitTask = cms.Task(
    ecalMultiFitUncalibRecHit
)
