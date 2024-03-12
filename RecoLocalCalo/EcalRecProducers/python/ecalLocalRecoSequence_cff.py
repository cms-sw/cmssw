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
ecalLocalRecoTask = cms.Task(ecalUncalibRecHit,ecalDetIdToBeRecovered,ecalRecHit,ecalPreshowerRecHit)
ecalLocalRecoSequence = cms.Sequence(ecalLocalRecoTask)
# foo bar baz
# JnJ2FDIT5zxNP
# 4h0S9uaZ4HGA4
