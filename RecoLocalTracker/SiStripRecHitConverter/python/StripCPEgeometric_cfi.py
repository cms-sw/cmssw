import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEgeometricESProducer = stripCPEESProducer.clone(
     ComponentName = 'StripCPEgeometric'
     ComponentType = 'StripCPEgeometric'
     parameters    = cms.PSet(
        TanDiffusionAngle            = 0.01,
        ThicknessRelativeUncertainty = 0.02,
        NoiseThreshold               = 2.3,
        MaybeNoiseThreshold          = 3.5,
        UncertaintyScaling           = 1.42,
        MinimumUncertainty           = 0.01
     )
)
