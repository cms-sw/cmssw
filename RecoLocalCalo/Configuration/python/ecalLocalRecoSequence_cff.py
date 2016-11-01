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

ecalUncalibRecHitSequence = cms.Sequence(ecalMultiFitUncalibRecHit*
                                        ecalDetIdToBeRecovered)

ecalRecHitSequence        = cms.Sequence(ecalRecHit*
                                         ecalCompactTrigPrim*
                                         ecalTPSkim+
                                         ecalPreshowerRecHit)

ecalLocalRecoSequence     = cms.Sequence(ecalUncalibRecHitSequence*
                                         ecalRecHitSequence)

from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHit_cfi import *
_phase2_timing_ecalRecHitSequence = cms.Sequence( ecalRecHitSequence.copy() + ecalDetailedTimeRecHit )
from Configuration.Eras.Modifier_phase2_timing_cff import phase2_timing
phase2_timing.toReplaceWith( ecalRecHitSequence, _phase2_timing_ecalRecHitSequence )
