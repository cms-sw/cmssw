import FWCore.ParameterSet.Config as cms

# FED size first designed for Pixels, but can be generalized to any detector
hltFEDSizeFilter = cms.EDFilter("HLTFEDSizeFilter",
   threshold = cms.int32(0),
   firstFED = cms.int32(0),
   lastFED = cms.int32(39),
   rawData = cms.InputTag("","","")
)
