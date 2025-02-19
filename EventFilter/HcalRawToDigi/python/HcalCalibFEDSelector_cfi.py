import FWCore.ParameterSet.Config as cms

hcalCalibFEDSelector = cms.EDProducer( "HcalCalibFEDSelector", 
   rawInputLabel   = cms.string( "source" ), 
   extraFEDsToKeep = cms.vint32( ) 
)
