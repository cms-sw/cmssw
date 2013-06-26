import FWCore.ParameterSet.Config as cms

hltPixelActivityFilter = cms.EDFilter( "HLTPixelActivityFilter",
   inputTag    = cms.InputTag( "hltSiPixelClusters" ),
   saveTags = cms.bool( False ),
   minClusters = cms.uint32( 3 ),
   maxClusters = cms.uint32( 0 )                                    
)
