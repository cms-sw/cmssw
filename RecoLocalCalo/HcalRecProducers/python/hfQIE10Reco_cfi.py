import FWCore.ParameterSet.Config as cms

hfQIE10Reco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis","HFQIE10DigiCollection"),
    Subdetector = cms.string('HFQIE10'),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(2),
    samplesToAdd = cms.int32(1),
    tsFromDB = cms.bool(True)
)


