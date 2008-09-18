import FWCore.ParameterSet.Config as cms

hltFEDSizeFilter = cms.EDFilter("HLTFEDSizeFilter",
   threshold = cms.untracked.int32(0),
   firstFED = cms.untracked.int32(0),
   lastFED = cms.untracked.int32(931),
   rawData = cms.InputTag("","","")
)
