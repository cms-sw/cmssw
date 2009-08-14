import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDFilter(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(1),
    Subdetector = cms.string('ZDC'),
    firstSample = cms.int32(3),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),

    #Tags for calculating status flags
    # None of the flag algorithms have been implemented for zdc, so these booleans do nothing
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True),
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False), # timing flags currently only implemented for HF
    
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
    ) # zdcreco


