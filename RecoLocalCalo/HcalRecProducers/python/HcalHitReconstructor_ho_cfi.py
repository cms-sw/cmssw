import FWCore.ParameterSet.Config as cms

horeco = cms.EDFilter(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(4),
    Subdetector = cms.string('HO'),
    firstSample = cms.int32(4),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),

    #Tags for calculating status flags
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True), # HSCP not implemented for horeco; this boolean does nothing
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False), # timing flags currently only implemented for HF
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
) # horeco


