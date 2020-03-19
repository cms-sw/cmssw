import FWCore.ParameterSet.Config as cms

# Calo geometry service model
#
# removed by tommaso
#
#ECAL conditions
#  include "CalibCalorimetry/EcalTrivialCondModules/data/EcalTrivialCondRetriever.cfi"
#
#TPG condition needed by ecalRecHit producer if TT recovery is ON
from RecoLocalCalo.EcalRecProducers.ecalRecHitTPGConditions_cff import *
#ECAL reconstruction
#from RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalCompactTrigPrim_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalTPSkim_cfi import *

from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi import *

#ecalUncalibRecHitSequence = cms.Sequence(ecalGlobalUncalibRecHit*
#                                         ecalDetIdToBeRecovered)

ecalUncalibRecHitTask = cms.Task(ecalMultiFitUncalibRecHit,
                                        ecalDetIdToBeRecovered)

ecalRecHitTask        = cms.Task(ecalRecHit,
                                         ecalCompactTrigPrim,
                                         ecalTPSkim,
                                         ecalPreshowerRecHit)

ecalLocalRecoTask     = cms.Task(ecalUncalibRecHitTask,
                                         ecalRecHitTask)

ecalUncalibRecHitSequence = cms.Sequence(ecalUncalibRecHitTask)
ecalRecHitSequence = cms.Sequence(ecalRecHitTask)
ecalLocalRecoSequence = cms.Sequence(ecalLocalRecoTask)


from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi import *
_phase2_timing_ecalRecHitTask = cms.Task( ecalRecHitTask.copy() , ecalDetailedTimeRecHit )
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith( ecalRecHitTask, _phase2_timing_ecalRecHitTask )

_fastSim_ecalRecHitTask = ecalRecHitTask.copyAndExclude([ecalCompactTrigPrim,ecalTPSkim])
_fastSim_ecalUncalibRecHitTask = ecalUncalibRecHitTask.copyAndExclude([ecalDetIdToBeRecovered])
from Configuration.Eras.Modifier_fastSim_cff import fastSim
fastSim.toReplaceWith(ecalRecHitTask, _fastSim_ecalRecHitTask)
fastSim.toReplaceWith(ecalUncalibRecHitTask, _fastSim_ecalUncalibRecHitTask)
