import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDProducer("ZdcSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(3),
    Subdetector = cms.string('ZDC'),
    firstSample = cms.int32(4),
    firstNoise = cms.int32(1),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    recoMethod = cms.int32(2)	
)


