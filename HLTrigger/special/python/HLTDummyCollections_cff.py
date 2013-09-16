import FWCore.ParameterSet.Config as cms

hltDummyEcalRawToRecHitFacility = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doEcal"),
  UnpackZDC = cms.bool(False),          # not actually needed here
  ESdigiCollection = cms.string( "" )   # not actually needed here
)

hltDummyHcalDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doHcal"),
  UnpackZDC = cms.bool(True),
  ESdigiCollection = cms.string( "" )   # not actually needed here
)

hltDummyEcalPreshowerDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doEcalPreshower"),
  UnpackZDC = cms.bool(False),          # not actually needed here     
  ESdigiCollection = cms.string( "" )
)

hltDummyMuonDTDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doMuonDT"),
  UnpackZDC = cms.bool(False),          # not actually needed here
  ESdigiCollection = cms.string( "" )   # not actually needed here
)

hltDummyMuonCSCDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doMuonCSC"),
  UnpackZDC = cms.bool(False),          # not actually needed here
  ESdigiCollection = cms.string( "" )   # not actually needed here
)

hltDummySiPixelDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doSiPixel"),
  UnpackZDC = cms.bool(False),          # not actually needed here
  ESdigiCollection = cms.string( "" )   # not actually needed here
)

hltDummySiStripRawToClustersFacility = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doSiStrip"),
  UnpackZDC = cms.bool(False),          # not actually needed here
  ESdigiCollection = cms.string( "" )   # not actually needed here
)

hltDummyGctDigis = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doGCT"),
  UnpackZDC = cms.bool(False),          # not actually needed here
  ESdigiCollection = cms.string( "" )   # not actually needed here
)


hltDummyL1GtObjectMap = cms.EDProducer("HLTDummyCollections",
  action = cms.string("doObjectMap"),
  UnpackZDC = cms.bool(False),          # not actually needed here
  ESdigiCollection = cms.string( "" )   # not actually needed here
)





