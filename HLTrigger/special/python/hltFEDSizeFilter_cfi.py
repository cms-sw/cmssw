import FWCore.ParameterSet.Config as cms

# FED size first designed for Pixels, but can be generalized to any detector
hltFEDSizeFilter = cms.EDFilter("HLTFEDSizeFilter",
   threshold = cms.uint32(0),
   firstFED = cms.uint32(0),
   lastFED = cms.uint32(39),
   rawData = cms.InputTag("","","")
)
