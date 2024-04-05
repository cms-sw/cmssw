import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Accelerators_cff import *

from RecoLocalCalo.EcalRecProducers.ecalPhase2DigiToPortableProducer_cfi import ecalPhase2DigiToPortableProducer as _ecalPhase2DigiToPortableProducer
ecalPhase2DigiToPortableProducer = _ecalPhase2DigiToPortableProducer.clone()

# portable weights
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2Portable_cfi import ecalUncalibRecHitPhase2Portable as _ecalUncalibRecHitPhase2Portable
ecalUncalibRecHitPhase2Portable = _ecalUncalibRecHitPhase2Portable.clone(
        digisLabelEB = 'ecalPhase2DigiToPortableProducer:ebDigis'
)

from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitSoAToLegacy_cfi import ecalUncalibRecHitSoAToLegacy as _ecalUncalibRecHitSoAToLegacy
ecalUncalibRecHitPhase2 = _ecalUncalibRecHitSoAToLegacy.clone(
    isPhase2 = True,
    uncalibRecHitsPortableEB = 'ecalUncalibRecHitPhase2Portable:EcalUncalibRecHitsEB',
    uncalibRecHitsPortableEE = None,
    recHitsLabelCPUEE = None
)


ecalUncalibRecHitPhase2Task = cms.Task(
  # convert phase2 digis to Portable Collection
  ecalPhase2DigiToPortableProducer, 
  # ECAL weights running on Portable
  ecalUncalibRecHitPhase2Portable,
  # Convert the uncalibrated rechits from Portable Collection to legacy format
  ecalUncalibRecHitPhase2
)
