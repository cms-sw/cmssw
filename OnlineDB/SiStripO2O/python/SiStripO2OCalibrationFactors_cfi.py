import FWCore.ParameterSet.Config as cms

SiStripO2OCalibrationFactors = cms.PSet(
    # Default values written when SiStrip objects are not found
    DefaultPedestal=cms.untracked.double(0),
    DefaultNoise=cms.untracked.double(51),
    DefaultThresholdLow=cms.untracked.double(2),
    DefaultThresholdHigh=cms.untracked.double(5),
    DefaultTickHeight=cms.untracked.double(690),
    DefaultAPVLatency=cms.untracked.uint32(142),
    DefaultAPVMode=cms.untracked.uint32(37),

    # Normalization Factor needed to convert Tick Height to Gain
    GainNormalizationFactor=cms.untracked.double(690),

    # Enable/Disable O2O of the following Objects
    UseAnalysis=cms.untracked.bool(False),
    UseFED=cms.untracked.bool(False),
    UseFEC=cms.untracked.bool(False),

    # Print additional debug information
    DebugMode=cms.untracked.bool(True)
    
    )

