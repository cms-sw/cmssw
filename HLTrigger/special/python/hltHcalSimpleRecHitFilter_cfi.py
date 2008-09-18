import FWCore.ParameterSet.Config as cms

hltHcalSimpleRecHITFilter = cms.EDFilter("HLTHcalSimpleRecHitFilter",
   threshold = cms.double(0),
   maskedChannels = cms.vint32(),
   HFRecHitCollection = cms.InputTag("","","")
)
