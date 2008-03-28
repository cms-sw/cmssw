import FWCore.ParameterSet.Config as cms

# Calo geometry service model
from Geometry.CaloEventSetup.CaloGeometry_cff import *
#
# removed by tommaso
#
#ECAL conditions
#  include "CalibCalorimetry/EcalTrivialCondModules/data/EcalTrivialCondRetriever.cfi"
#ECAL reconstruction
from RecoLocalCalo.EcalRecProducers.ecalWeightUncalibRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalRecHit_cfi import *
from RecoLocalCalo.EcalRecProducers.ecalPreshowerRecHit_cfi import *
ecalLocalRecoSequence = cms.Sequence(ecalWeightUncalibRecHit*ecalRecHit+ecalPreshowerRecHit)
ecalLocalRecoSequence_nopreshower = cms.Sequence(ecalWeightUncalibRecHit*ecalRecHit)

