import FWCore.ParameterSet.Config as cms

hiClusterCompatibility = cms.EDProducer("ClusterCompatibilityProducer",
   inputTag      = cms.InputTag( "siPixelRecHits" ),
   saveTags = cms.bool( False ),
   minZ          = cms.double(-40.0),
   maxZ          = cms.double(40.05),
   zStep         = cms.double(0.2)
)
