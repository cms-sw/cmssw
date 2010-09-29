import FWCore.ParameterSet.Config as cms

hltPixelActivityFilter = cms.EDFilter( "HLTPixelActivityFilter",
   inputTag    = cms.InputTag( "hltSiPixelClusters" ),
   saveTag     = cms.untracked.bool( False ),
   minClusters = cms.uint32( 3 ),
   maxClusters = cms.uint32( 0 )                                    
)
