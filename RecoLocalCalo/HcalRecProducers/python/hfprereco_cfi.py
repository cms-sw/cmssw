import FWCore.ParameterSet.Config as cms

hfprereco = cms.EDProducer("HFPreReconstructor",
    digiLabel = cms.InputTag("simHcalUnsuppressedDigis", "HFQIE10DigiCollection"),
    dropZSmarkedPassed = cms.bool(True),
    firstSample = cms.int32(2),
    tsFromDB = cms.bool(True)
)
