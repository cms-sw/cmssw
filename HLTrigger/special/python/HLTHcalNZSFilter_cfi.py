import FWCore.ParameterSet.Config as cms

hltHcalNZSFilter = cms.EDFilter( "HLTHcalNZSFilter", 
   InputLabel    = cms.string( "source" ),
   FilterSummary = cms.untracked.bool( False ) 
)
