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
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cff import *
ecalRecHit.cpu.EBuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
ecalRecHit.cpu.EEuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'
ecalRecHit.cpu.ChannelStatusToBeExcluded = [
    'kDAC',
    'kNoLaser',
    'kNoisy',
    'kNNoisy',  
    'kFixedG6',
    'kFixedG1',
    'kFixedG0',
    'kNonRespondingIsolated',
    'kDeadVFE',
    'kDeadFE',
    'kNoDataNoTP'
]
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *

ecalLocalRecoTaskCosmics = cms.Task(
    ecalFixedAlphaBetaFitUncalibRecHit,
    ecalWeightUncalibRecHit,
    ecalDetIdToBeRecovered,
    ecalCalibratedRecHitTask,
    ecalPreshowerRecHit
)
ecalLocalRecoSequenceCosmics = cms.Sequence(ecalLocalRecoTaskCosmics)
