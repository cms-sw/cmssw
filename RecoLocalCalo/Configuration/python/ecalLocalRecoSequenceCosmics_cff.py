import FWCore.ParameterSet.Config as cms

# Calo geometry service model
#
# removed by tommaso
#
#ECAL conditions
#  include "CalibCalorimetry/EcalTrivialCondModules/data/EcalTrivialCondRetriever.cfi"
#ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalFixedAlphaBetaFitUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
ecalLocalRecoSequence = cms.Sequence(ecalFixedAlphaBetaFitUncalibRecHit*ecalWeightUncalibRecHit*ecalRecHit+ecalPreshowerRecHit)
ecalRecHit.EBuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEB'
ecalRecHit.EEuncalibRecHitCollection = 'ecalFixedAlphaBetaFitUncalibRecHit:EcalUncalibRecHitsEE'
ecalRecHit.ChannelStatusToBeExcluded = [1]

