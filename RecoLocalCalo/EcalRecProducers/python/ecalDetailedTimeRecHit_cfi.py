import FWCore.ParameterSet.Config as cms

from SimCalorimetry.EcalSimProducers.ecalTimeDigiParameters_cff import *
from RecoLocalCalo.EcalRecProducers.ecalDetailedTimeRecHitProducer_cfi import ecalDetailedTimeRecHitProducer
ecalDetailedTimeRecHit = ecalDetailedTimeRecHitProducer.clone(
    EBTimeLayer = ecal_time_digi_parameters.timeLayerBarrel,
    EETimeLayer = ecal_time_digi_parameters.timeLayerEndcap
)
