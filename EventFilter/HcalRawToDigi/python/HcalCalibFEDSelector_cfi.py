import FWCore.ParameterSet.Config as cms

hcalCalibFEDSelector = cms.EDProducer( "HcalCalibFEDSelector", 
   rawInputLabel   = cms.string( "rawDataCollector" ), 
   extraFEDsToKeep = cms.vint32( ) 
)
