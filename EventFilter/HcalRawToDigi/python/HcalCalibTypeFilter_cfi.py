import FWCore.ParameterSet.Config as cms

hcalCalibTypeFilter = cms.EDFilter( "HcalCalibTypeFilter", 
   InputLabel    = cms.string( "rawDataCollector" ),
   CalibTypes    = cms.vint32( 1,2,3,4,5 ),
   FilterSummary = cms.untracked.bool( False ) 
)
