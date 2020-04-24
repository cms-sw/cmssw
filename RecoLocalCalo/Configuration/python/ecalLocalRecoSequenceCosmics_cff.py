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
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
ecalLocalRecoSequenceCosmics = cms.Sequence(ecalFixedAlphaBetaFitUncalibRecHit*ecalWeightUncalibRecHit*ecalDetIdToBeRecovered*ecalRecHit+ecalPreshowerRecHit)
ecalRecHit.EBuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
ecalRecHit.EEuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'
ecalRecHit.ChannelStatusToBeExcluded = [
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
