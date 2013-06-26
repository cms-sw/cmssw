import FWCore.ParameterSet.Config as cms

StripCPEgeometricESProducer =cms.ESProducer("StripCPEESProducer",
                                            ComponentName = cms.string('StripCPEgeometric'),
                                            ComponentType = cms.string('StripCPEgeometric'),
                                            TanDiffusionAngle            = cms.double(0.01),
                                            ThicknessRelativeUncertainty = cms.double(0.02),
                                            NoiseThreshold               = cms.double(2.3),
                                            MaybeNoiseThreshold          = cms.double(3.5),
                                            UncertaintyScaling           = cms.double(1.42),
                                            MinimumUncertainty           = cms.double(0.01),
)
