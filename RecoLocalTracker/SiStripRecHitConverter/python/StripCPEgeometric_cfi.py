import FWCore.ParameterSet.Config as cms

StripCPEgeometricESProducer =cms.ESProducer("StripCPEESProducer",
               ComponentName = cms.string('StripCPEgeometric'),
               TanDiffusionAngle            = cms.double(0.01),
               ThicknessRelativeUncertainty = cms.double(0.02),
               NoiseThreshold               = cms.double(2.3),
               MaybeNoiseThreshold          = cms.double(3.2),
               UncertaintyScaling           = cms.double(1.42),
               #---Crosstalk
               APVpeakmode             = cms.bool(False),
               # Deconvolution Mode
               CouplingConstantDecTIB  = cms.double(0.12),
               CouplingConstantDecTID  = cms.double(0.12),
               CouplingConstantDecTOB  = cms.double(0.12),
               CouplingConstantDecTEC  = cms.double(0.12),
               # Peak Mode
               CouplingConstantPeakTIB = cms.double(0.006),
               CouplingConstantPeakTOB = cms.double(0.04),
               CouplingConstantPeakTID = cms.double(0.03),
               CouplingConstantPeakTEC = cms.double(0.03),
)
