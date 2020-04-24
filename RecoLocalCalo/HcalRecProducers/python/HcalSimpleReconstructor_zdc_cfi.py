import FWCore.ParameterSet.Config as cms

zdcreco = cms.EDProducer("ZdcSimpleReconstructor",
    correctionPhaseNS = cms.double(0.0),
    digiLabelhcal = cms.InputTag("hcalDigis"),
    digiLabelcastor = cms.InputTag("castorDigis"),
    digiLabelQIE10ZDC = cms.InputTag("hcalDigis:ZDC"),
    Subdetector = cms.string('ZDC'),
    correctForPhaseContainment = cms.bool(False),
    correctForTimeslew = cms.bool(False),
    dropZSmarkedPassed = cms.bool(True),
    recoMethod = cms.int32(2)	
)


