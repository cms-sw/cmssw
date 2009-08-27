import FWCore.ParameterSet.Config as cms

hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter", 
   InputTag      = cms.InputTag( "source" ),
   CalibTypes    = cms.vint32( 1,2,3,4,5 ),
   FilterSummary = cms.untracked.bool( False ) 
)
