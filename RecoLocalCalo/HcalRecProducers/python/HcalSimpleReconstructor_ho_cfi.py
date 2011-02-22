import FWCore.ParameterSet.Config as cms

horeco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(13.0),
    digiLabel = cms.InputTag("hcalDigis"),
    Subdetector = cms.string('HO'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True)
)


