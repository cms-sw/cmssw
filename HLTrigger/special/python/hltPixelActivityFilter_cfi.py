import FWCore.ParameterSet.Config as cms

hltPixelActivityFilter = cms.EDFilter( "HLTPixelActivityFilter",
   pixelTag     = cms.InputTag( "hltSiPixelClusters" ),
   minClusters  = cms.uint32( 3 )
)
