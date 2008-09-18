import FWCore.ParameterSet.Config as cms

hltHcalSimpleRecHITFilter = cms.EDFilter("HLTHcalSimpleRecHitFilter",
   threshold = cms.untracked.double(0),
   maskedChannels = cms.untracked.vint32(0,0),
   HFRecHitCollection = cms.InputTag("","","")
)
