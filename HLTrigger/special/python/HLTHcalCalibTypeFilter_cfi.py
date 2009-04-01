import FWCore.ParameterSet.Config as cms

hltHcalCalibTypeFilter = cms.EDFilter( "HLTHcalCalibTypeFilter", 
   InputLabel    = cms.string( "source" ),
   CalibTypes    = cms.vint32( 1,2,3,4,5 ),
   FilterSummary = cms.bool( False ) 
)
