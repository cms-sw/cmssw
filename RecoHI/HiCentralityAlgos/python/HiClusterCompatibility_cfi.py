import FWCore.ParameterSet.Config as cms

hiClusterCompatibility = cms.EDProducer("ClusterCompatibilityProducer",
   inputTag      = cms.InputTag( "siPixelRecHits" ),
   minZ          = cms.double(-40.0),
   maxZ          = cms.double(40.05),
   zStep         = cms.double(0.2)
)
# foo bar baz
# 9Cbz3VeYtea3l
