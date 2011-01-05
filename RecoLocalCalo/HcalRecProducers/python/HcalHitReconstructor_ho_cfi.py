import FWCore.ParameterSet.Config as cms

horeco = cms.EDProducer(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HO'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),

    # Set offset between firstSample value and
    # first sample to be stored in aux word
    firstAuxOffset = cms.int32(0),

    #Tags for calculating status flags
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True), # HSCP not implemented for horeco; this boolean does nothing
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False), # timing flags currently only implemented for HF
    setPulseShapeFlags = cms.bool(False),
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
) # horeco


