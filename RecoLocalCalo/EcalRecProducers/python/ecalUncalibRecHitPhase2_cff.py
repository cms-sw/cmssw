import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.CUDACore.SwitchProducerCUDA import SwitchProducerCUDA

# ECAL Phase 2 weights running on CPU
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2_cfi import ecalUncalibRecHitPhase2 as _ecalUncalibRecHitPhase2
ecalUncalibRecHitPhase2CPU = _ecalUncalibRecHitPhase2.clone() 
ecalUncalibRecHitPhase2 = SwitchProducerCUDA(
  cpu = ecalUncalibRecHitPhase2CPU
)

ecalUncalibRecHitPhase2Task = cms.Task(
        # ECAL weights running on CPU
        ecalUncalibRecHitPhase2
)


from Configuration.StandardSequences.Accelerators_cff import *

# process modifier to run alpaka implementation
from Configuration.ProcessModifiers.alpaka_cff import alpaka

#ECAL Phase 2 Digis Producer running on the accelerator
from RecoLocalCalo.EcalRecProducers.ecalPhase2DigiToPortableProducer_cfi import ecalPhase2DigiToPortableProducer as _ecalPhase2DigiToPortableProducer
ecalPhase2DigiToPortableProducer = _ecalPhase2DigiToPortableProducer.clone()

# ECAL Phase 2 weights portable running
from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitPhase2Portable_cfi import ecalUncalibRecHitPhase2Portable as _ecalUncalibRecHitPhase2Portable
ecalUncalibRecHitPhase2Portable = _ecalUncalibRecHitPhase2Portable.clone(
        digisLabelEB = 'ecalPhase2DigiToPortableProducer:ebDigis'
)

from RecoLocalCalo.EcalRecProducers.ecalUncalibRecHitSoAToLegacy_cfi import ecalUncalibRecHitSoAToLegacy as _ecalUncalibRecHitSoAToLegacy
alpaka.toModify(ecalUncalibRecHitPhase2,
    cpu = _ecalUncalibRecHitSoAToLegacy.clone(
    isPhase2 = True,
    uncalibRecHitsPortableEB = 'ecalUncalibRecHitPhase2Portable:EcalUncalibRecHitsEB',
    uncalibRecHitsPortableEE = None,
    recHitsLabelCPUEE = None
    )
)


alpaka.toReplaceWith(ecalUncalibRecHitPhase2Task, cms.Task(
  # convert phase2 digis to Portable Collection
  ecalPhase2DigiToPortableProducer, 
  # ECAL weights running on Portable
  ecalUncalibRecHitPhase2Portable,
  # Convert the uncalibrated rechits from Portable Collection to legacy format
  ecalUncalibRecHitPhase2
))
