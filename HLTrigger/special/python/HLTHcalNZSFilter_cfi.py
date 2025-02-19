import FWCore.ParameterSet.Config as cms

hltHcalNZSFilter = cms.EDFilter( "HLTHcalNZSFilter", 
   InputTag      = cms.InputTag( "source" ),
   FilterSummary = cms.untracked.bool( False ),
   saveTags = cms.bool( False )
)
