import FWCore.ParameterSet.Config as cms


pvMonitor = cms.EDAnalyzer("PrimaryVertexMonitor",
   TopFolderName  = cms.string("OfflinePV"),
   AlignmentLabel = cms.string("Alignment"),                           
   vertexLabel    = cms.InputTag("offlinePrimaryVertices"),
   beamSpotLabel  = cms.InputTag("offlineBeamSpot"),
   ndof           = cms.int32( 4 ),                           
   TkSizeBin      = cms.int32( 100  ),
   TkSizeMax      = cms.double(499.5),                       
   TkSizeMin      = cms.double( -0.5),
   Xpos           = cms.double(0.1),
   Ypos           = cms.double(0.0),
   DxyBin         = cms.int32(100),
   DxyMax         = cms.double(5000.0),
   DxyMin         = cms.double(-5000.0),                        
   DzBin          = cms.int32(100),
   DzMax          = cms.double(2000.0),
   DzMin          = cms.double(-2000.0),                                             
   PhiBin         = cms.int32(32),
   PhiMax         = cms.double(3.141592654),
   PhiMin         = cms.double(-3.141592654),
   EtaBin         = cms.int32(26),
   EtaMax         = cms.double(2.5),
   EtaMin         = cms.double(-2.5)
)
