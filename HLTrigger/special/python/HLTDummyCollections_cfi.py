import FWCore.ParameterSet.Config as cms

hltDummyEcalRawToRecHitFacility = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doEcal")
)

hltDummyHcalDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doHcal"),
  UnpackZDC = cms.bool(True)
)

hltDummyEcalPreshowerDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doEcalPreshower"),
  ESdigiCollection = cms.string( "" )
)

hltDummyMuonDTDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doMuonDT")
)

hltDummyMuonCSCDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doMuonCSC")
)

hltDummySiPixelDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doSiPixel")
)

hltDummySiStripRawToClustersFacility = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doSiStrip")
)

hltDummyGctDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doGCT")
)


hltDummyL1GtObjectMap = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doObjectMap")
)





