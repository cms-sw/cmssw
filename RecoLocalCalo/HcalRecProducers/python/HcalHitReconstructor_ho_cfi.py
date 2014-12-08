import FWCore.ParameterSet.Config as cms

horeco = cms.EDProducer(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HO'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(4),
    tsFromDB = cms.bool(True),
    recoParamsFromDB = cms.bool(True),
    useLeakCorrection = cms.bool(False),
    puCorrMethod = cms.int32(0),

    # Set time slice for first digi to be stored in aux word
    # (HO uses time slices 4-7)
    firstAuxTS = cms.int32(4),

    #Tags for calculating status flags
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True), # HSCP not implemented for horeco; this boolean does nothing
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False), # timing flags currently only implemented for HF
    setPulseShapeFlags = cms.bool(False),  # not yet defined for HO
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
) # horeco


