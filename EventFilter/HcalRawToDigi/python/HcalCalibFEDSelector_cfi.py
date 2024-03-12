import FWCore.ParameterSet.Config as cms

hcalCalibFEDSelector = cms.EDProducer( "HcalCalibFEDSelector", 
   rawInputLabel   = cms.string( "rawDataCollector" ), 
   extraFEDsToKeep = cms.vint32( ) 
)
# foo bar baz
# I2Zx68mR1IvfY
