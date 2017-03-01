import FWCore.ParameterSet.Config as cms

from RecoLocalTracker.SiStripRecHitConverter.StripCPEESProducer_cfi import *
StripCPEgeometricESProducer = stripCPEESProducer.clone()
StripCPEgeometricESProducer.ComponentName = cms.string('StripCPEgeometric')
StripCPEgeometricESProducer.ComponentType = cms.string('StripCPEgeometric')
StripCPEgeometricESProducer.parameters    = cms.PSet(
   TanDiffusionAngle            = cms.double(0.01),
   ThicknessRelativeUncertainty = cms.double(0.02),
   NoiseThreshold               = cms.double(2.3),
   MaybeNoiseThreshold          = cms.double(3.5),
   UncertaintyScaling           = cms.double(1.42),
   MinimumUncertainty           = cms.double(0.01)
)
