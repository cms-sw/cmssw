import  RecoLocalCalo.EcalRecProducers.ecalMultiFitUncalibRecHitProducer_cfi as _mod

# producer of rechits starting from digis
ecalMultiFitUncalibRecHit = _mod.ecalMultiFitUncalibRecHitProducer.clone()

# use CC timing method for Run3 and Phase 2 (carried over from Run3 era)
import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.ecal_cctiming_cff import ecal_cctiming
ecal_cctiming.toModify(ecalMultiFitUncalibRecHit,
    algoPSet = dict(timealgo = 'crossCorrelationMethod',
        EBtimeNconst = 25.5,
        EBtimeConstantTerm = 0.85,
        outOfTimeThresholdGain12pEB = 3.0,
        outOfTimeThresholdGain12mEB = 3.0,
        outOfTimeThresholdGain61pEB = 12.0,
        outOfTimeThresholdGain61mEB = 12.0,
        timeCalibTag = ':CC',
        timeOffsetTag = ':CC'
    )
)

# this overrides the modifications made by the ecal_cctiming modifier if both modifiers are active
from Configuration.ProcessModifiers.gpuValidationEcal_cff import gpuValidationEcal
gpuValidationEcal.toModify(ecalMultiFitUncalibRecHit,
    algoPSet = dict(timealgo = 'RatioMethod',
        outOfTimeThresholdGain12pEB = 5.,
        outOfTimeThresholdGain12mEB = 5.,
        outOfTimeThresholdGain61pEB = 5.,
        outOfTimeThresholdGain61mEB = 5.,
        timeCalibTag = ':',
        timeOffsetTag = ':'
    )
)

