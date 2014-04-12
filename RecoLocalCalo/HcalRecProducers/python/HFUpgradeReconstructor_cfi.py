import FWCore.ParameterSet.Config as cms

hfUpgradeReco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis","HFUpgradeDigiCollection"),
    Subdetector = cms.string('upgradeHF'),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(2),
    samplesToAdd = cms.int32(1),
    tsFromDB = cms.bool(True)
)


