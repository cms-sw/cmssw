import FWCore.ParameterSet.Config as cms

hltHcalSimpleRecHitFilter = cms.EDFilter("HLTHcalSimpleRecHitFilter",
   threshold = cms.double(3.0),
   minNHitsNeg = cms.int32(1),
   minNHitsPos = cms.int32(1),
   doCoincidence = cms.bool(True),
   maskedChannels = cms.vuint32(), # now by raw detid, not hashed id
   HFRecHitCollection = cms.InputTag("hltHfreco"),
   saveTags = cms.bool( False )
)
