import FWCore.ParameterSet.Config as cms

horeco = cms.EDFilter(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalZeroSuppressedDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HO'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    #Tags for calculating status flags
    correctTiming = cms.bool(True),
    # Noise, HSCP flags don't exist for HO yet, so setting these booleans true does nothing
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True), 
    # Check ADC saturation
    setSaturationFlags = cms.bool(True),
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
) #horeco


