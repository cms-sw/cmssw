import FWCore.ParameterSet.Config as cms

hfprereco = cms.EDProducer("HFPreReconstructor",
    digiLabel = cms.InputTag("hcalDigis"),
    dropZSmarkedPassed = cms.bool(True),
    tsFromDB = cms.bool(False),
    sumAllTimeSlices = cms.bool(False)
)
