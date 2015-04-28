import FWCore.ParameterSet.Config as cms

hiClusterCompatibility = cms.EDProducer("ClusterCompatibilityProducer",
   inputTag      = cms.InputTag( "hltSiPixelRecHits" ),
   saveTags = cms.bool( False ),
   minZ          = cms.double(-20.0),
   maxZ          = cms.double(20.05),
   zStep         = cms.double(0.2)
)
