import FWCore.ParameterSet.Config as cms

# re-calibrated rechit producer
from RecoLocalCalo.EcalRecProducers.ecalRecalibRecHitProducer_cfi import ecalRecalibRecHitProducer
ecalRecHit = ecalRecalibRecHitProducer.clone()
