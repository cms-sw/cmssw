import FWCore.ParameterSet.Config as cms

# Calo geometry service model
#ECAL conditions
#
# removed : this goes into CalibCalorimetry/Configuration/data/Ecal_FakeCalibrations.cff
#
#  include "CalibCalorimetry/EcalTrivialCondModules/data/EcalTrivialCondRetriever.cfi"
#
#ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalDetIdToBeRecovered_cfi import *
ecalLocalRecoSequence = cms.Sequence(ecalUncalibRecHit*ecalDetIdToBeRecovered*ecalRecHit*ecalPreshowerRecHit)
