import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDFilter("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalDigis"),
    samplesToAdd = cms.int32(1),
    Subdetector = cms.string('ZDC'),
    firstSample = cms.int32(3),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False)
)


