import FWCore.ParameterSet.Config as cms

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
pvMonitor = DQMEDAnalyzer('PrimaryVertexMonitor',
   TopFolderName  = cms.string("OfflinePV"),
   AlignmentLabel = cms.string("Alignment"),                           
   vertexLabel    = cms.InputTag("offlinePrimaryVertices"),
   beamSpotLabel  = cms.InputTag("offlineBeamSpot"),
   useHPforAlignmentPlots = cms.bool(True),
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
   PhiBin2D       = cms.int32(12),
   PhiMax         = cms.double(3.141592654),
   PhiMin         = cms.double(-3.141592654),
   EtaBin         = cms.int32(26),
   EtaBin2D       = cms.int32(8),
   EtaMax         = cms.double(2.5),
   EtaMin         = cms.double(-2.5)
)

# same as above, should be in sync with cut used in Vertex finder...
from Configuration.Eras.Modifier_phase1Pixel_cff import phase1Pixel
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.Eras.Modifier_phase2_tracker_cff import phase2_tracker
phase1Pixel.toModify(pvMonitor, EtaBin=28, EtaMin=-2.7, EtaMax=2.7)
run3_common.toModify(pvMonitor, Xpos = 0.15, Ypos=-0.15)  #recentering since initial Run3 beamspot is at (0.17,-0.18) cm
phase2_tracker.toModify(pvMonitor, EtaBin=41, EtaBin2D=9, EtaMin=-4.0, EtaMax=4.0)

