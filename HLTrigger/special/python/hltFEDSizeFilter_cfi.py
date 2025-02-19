import FWCore.ParameterSet.Config as cms

# FED size first designed for Pixels, but can be generalized to any detector
hltFEDSizeFilter = cms.EDFilter("HLTFEDSizeFilter",
   rawData = cms.InputTag("source","",""),  # RAW data
   threshold = cms.uint32(0),           # 0 is pass-through, 1 means "FED ispresent", higher values are just FED size
   firstFED = cms.uint32(0),            # first FED, inclusive
   lastFED = cms.uint32(39),            # last FED, inclusive
   requireAllFEDs = cms.bool(False),     # is True, *all* FEDs must be above threshold; if False, only *one* is required
   saveTags = cms.bool( False )
)
