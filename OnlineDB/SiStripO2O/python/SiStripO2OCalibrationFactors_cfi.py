import FWCore.ParameterSet.Config as cms

SiStripO2OCalibrationFactors = cms.PSet(
    # Default values written when SiStrip objects are not found
    DefaultPedestal=cms.untracked.double(0),
    DefaultNoise=cms.untracked.double(51),
    DefaultThresholdLow=cms.untracked.double(2),
    DefaultThresholdHigh=cms.untracked.double(5),
    DefaultTickHeight=cms.untracked.double(690),

    # Normalization Factor needed to convert Tick Height to Gain
    GainNormalizationFactor=cms.untracked.double(690)
    
    )

