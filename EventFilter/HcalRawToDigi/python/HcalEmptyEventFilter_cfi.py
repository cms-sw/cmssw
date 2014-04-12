import FWCore.ParameterSet.Config as cms

hcalEmptyEventFilter = cms.EDFilter( "HcalEmptyEventFilter", 
   InputLabel    = cms.InputTag( "rawDataCollector" )
)
