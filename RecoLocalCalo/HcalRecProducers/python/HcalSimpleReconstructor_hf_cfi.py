import FWCore.ParameterSet.Config as cms

hfreco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HF'),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True)
)


