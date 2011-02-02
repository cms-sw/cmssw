import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDProducer(
    "ZdcHitReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(3),
    Subdetector = cms.string('ZDC'),
    firstSample = cms.int32(4),
    firstNoise = cms.int32(1),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    recoMethod = cms.int32(2),

    # Set offset between firstSample value and
    # first sample to be stored in aux word
    firstAuxOffset = cms.int32(0),
        
    #Tags for calculating status flags
    # None of the flag algorithms have been implemented for zdc, so these booleans do nothing
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True),
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False), # timing flags currently only implemented for HF
    
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
    ) # zdcreco


