import FWCore.ParameterSet.Config as cms

hltDummyEcalRawToRecHitFacility = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doEcal")
)

hltDummyHcalDigis = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doHcal"),
  UnpackZDC = cms.bool(True)
)

hltDummyEcalPreshowerDigis = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doEcalPreshower"),
  ESdigiCollection = cms.string( "" )
)

hltDummyMuonDTDigis = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doMuonDT")
)

hltDummyMuonCSCDigis = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doMuonCSC")
)

hltDummySiPixelDigis = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doSiPixel")
)

hltDummySiStripRawToClustersFacility = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doSiStrip")
)

hltDummyGctDigis = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doGCT")
)


hltDummyL1GtObjectMap = cms.EDFilter("HLTDummyCollections",
  action = cms.string("doObjectMap")
)





