import FWCore.ParameterSet.Config as cms

StripCPEgeometricESProducer =cms.ESProducer("StripCPEESProducer",
               ComponentName = cms.string('StripCPEgeometric'),
               APVpeakmode             = cms.bool(False),
               TanDriftAngle           = cms.double(0.01),
               #---Crosstalk
               CouplingConstantSpread  = cms.double(0.055),
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


