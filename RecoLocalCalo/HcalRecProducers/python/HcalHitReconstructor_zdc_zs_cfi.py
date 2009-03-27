import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDFilter(
    "HcalHitReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalZeroSuppressedDigis"),
    samplesToAdd = cms.int32(1),
    Subdetector = cms.string('ZDC'),
    firstSample = cms.int32(3),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),

    #Tags for calculating status flags
    # None of the flag algorithms have been implemented for ZDC, so these booleans do nothing
    correctTiming = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setHSCPFlags  = cms.bool(True),
    setSaturationFlags = cms.bool(True),
    saturationParameters=  cms.PSet(maxADCvalue=cms.int32(127))
                       
)


