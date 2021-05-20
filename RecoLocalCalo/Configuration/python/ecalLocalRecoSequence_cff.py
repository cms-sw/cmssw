import FWCore.ParameterSet.Config as cms

# TPG condition needed by ecalRecHit producer if TT recovery is ON
from RecoLocalCalo.EcalRecProducers.ecalRecHitTPGConditions_cff import *

# ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cff import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cff import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalCompactTrigPrim_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalTPSkim_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi import *

ecalUncalibRecHitTask = cms.Task(
    ecalMultiFitUncalibRecHitTask,
    ecalDetIdToBeRecovered)

ecalUncalibRecHitSequence = cms.Sequence(ecalUncalibRecHitTask)

ecalRecHitNoTPTask = cms.Task(
    ecalCalibratedRecHitTask,
    ecalPreshowerRecHit)

ecalRecHitNoTPSequence = cms.Sequence(ecalRecHitNoTPTask)

ecalRecHitTask = cms.Task(
    ecalCompactTrigPrim,
    ecalTPSkim,
    ecalRecHitNoTPTask)

ecalRecHitSequence = cms.Sequence(ecalRecHitTask)

ecalLocalRecoTask = cms.Task(
    ecalUncalibRecHitTask,
    ecalRecHitTask)

ecalLocalRecoSequence = cms.Sequence(ecalLocalRecoTask)

ecalOnlyLocalRecoTask = cms.Task(
    ecalUncalibRecHitTask,
    ecalRecHitNoTPTask)

ecalOnlyLocalRecoSequence = cms.Sequence(ecalOnlyLocalRecoTask)

# Phase 2 modifications
from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi import *
_phase2_timing_ecalRecHitTask = cms.Task( ecalRecHitTask.copy() , ecalDetailedTimeRecHit )
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith( ecalRecHitTask, _phase2_timing_ecalRecHitTask )

# FastSim modifications
_fastSim_ecalRecHitTask = ecalRecHitTask.copyAndExclude([ecalCompactTrigPrim,ecalTPSkim])
_fastSim_ecalUncalibRecHitTask = ecalUncalibRecHitTask.copyAndExclude([ecalDetIdToBeRecovered])
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(ecalRecHitTask, _fastSim_ecalRecHitTask)
fastSim.toReplaceWith(ecalUncalibRecHitTask, _fastSim_ecalUncalibRecHitTask)
