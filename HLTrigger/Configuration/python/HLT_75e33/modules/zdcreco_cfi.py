import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDProducer("ZdcHitReconstructor",
    AuxTSvec = cms.vint32(4, 5, 6, 7),
    Subdetector = cms.string('ZDC'),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    correctTiming = cms.bool(True),
    correctionPhaseNS = cms.double(0.0),
    digiLabelQIE10ZDC = cms.InputTag("hcalDigis","ZDC"),
    digiLabelcastor = cms.InputTag("castorDigis"),
    digiLabelhcal = cms.InputTag("hcalDigis"),
    dropZSmarkedPassed = cms.bool(True),
    lowGainFrac = cms.double(8.15),
    lowGainOffset = cms.int32(1),
    recoMethod = cms.int32(2),
    saturationParameters = cms.PSet(
        maxADCvalue = cms.int32(127)
    ),
    setHSCPFlags = cms.bool(True),
    setNoiseFlags = cms.bool(True),
    setSaturationFlags = cms.bool(True),
    setTimingTrustFlags = cms.bool(False)
)
