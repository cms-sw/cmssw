import FWCore.ParameterSet.Config as cms

hbheUpgradeReco = cms.EDProducer("HcalSimpleReconstructor",
    correctionPhaseNS = cms.double(-5.0),  
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis","HBHEUpgradeDigiCollection"),
    Subdetector = cms.string('upgradeHBHE'),
    correctForPhaseContainment = cms.bool(True),
    correctForTimeslew = cms.bool(True),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(4),
    samplesToAdd = cms.int32(2),
    tsFromDB = cms.bool(True)
)

