import FWCore.ParameterSet.Config as cms

hfprereco = cms.EDProducer("HFPreReconstructor",
    digiLabel = cms.InputTag("hcalDigis", "HFQIE10DigiCollection"),
    dropZSmarkedPassed = cms.bool(True),
    tsFromDB = cms.bool(False)
)
