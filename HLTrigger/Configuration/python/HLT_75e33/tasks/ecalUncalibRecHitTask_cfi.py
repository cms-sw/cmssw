import FWCore.ParameterSet.Config as cms

from ..modules.hltEcalDetIdToBeRecovered_cfi import *
from ..tasks.ecalMultiFitUncalibRecHitTask_cfi import *

ecalUncalibRecHitTask = cms.Task(
    hltEcalDetIdToBeRecovered,
    ecalMultiFitUncalibRecHitTask
)
# foo bar baz
# VaZUDVcsE3cb2
# xwxKuxXMvezJN
