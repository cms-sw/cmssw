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
from RecoLocalCalo.EcalRecProducers.ecalGlobalUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
ecalLocalRecoSequence = cms.Sequence(ecalGlobalUncalibRecHit*ecalDetIdToBeRecovered*ecalRecHit+ecalPreshowerRecHit)
ecalLocalRecoSequence_nopreshower = cms.Sequence(ecalGlobalUncalibRecHit*ecalRecHit)
ecalRecHit.ChannelStatusToBeExcluded = [ 10, 11, 12, 13, 14, 78, 142 ]
