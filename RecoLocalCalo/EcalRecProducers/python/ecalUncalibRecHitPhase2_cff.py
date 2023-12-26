#alpaka with no switch producer
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Accelerators_cff import *
from HeterogeneousCore.AlpakaCore.ProcessAcceleratorAlpaka_cfi import ProcessAcceleratorAlpaka

from RecoLocalCalo.EcalRecProducers.ecalPhase2DigiToPortableProducer_cfi import ecalPhase2DigiToPortableProducer as _ecalPhase2DigiToPortableProducer
ecalPhase2DigiToPortableProducer = _ecalPhase2DigiToPortableProducer.clone()

# portable weights
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2Portable_cfi import ecalUncalibRecHitPhase2Portable as _ecalUncalibRecHitPhase2Portable
ecalUncalibRecHitPhase2Portable = _ecalUncalibRecHitPhase2Portable.clone(
  digisLabelEB = ('ecalPhase2DigiToPortableProducer', 'ebDigis')
)

from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitConvertPortable2CPUFormat_cfi import ecalUncalibRecHitConvertPortable2CPUFormat as _ecalUncalibRecHitConvertPortable2CPUFormat
ecalUncalibRecHitPhase2 = _ecalUncalibRecHitConvertPortable2CPUFormat.clone(
    isPhase2 = True,
    uncalibratedRecHitsLabelPortableEB = ('ecalUncalibRecHitPhase2Portable', 'EcalUncalibRecHitsEB'),
    uncalibratedRecHitsLabelPortableEE = None,
    uncalibratedRecHitsLabelCPUEE = None
)


ecalUncalibRecHitPhase2Task = cms.Task(
  # convert phase2 digis to Portable Collection
  ecalPhase2DigiToPortableProducer, 
  # ECAL weights running on Portable
  ecalUncalibRecHitPhase2Portable,
  # Convert the uncalibrated rechits from Portable Collection to legacy format
  ecalUncalibRecHitPhase2
)
