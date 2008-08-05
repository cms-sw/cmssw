import FWCore.ParameterSet.Config as cms

#-----------------------
#  Reconstruction Modules
#-----------------------
# Real data raw to digi
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'source'
# Local and Track Reconstruction
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
CTF_P5_MeasurementTracker.pixelClusterProducer = ''
RS_P5_MeasurementTracker.pixelClusterProducer = ''
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

#-----------------------
#  SiStripMonitorTrack
#-----------------------
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack.TrackProducer       = cms.string('ctfWithMaterialTracksP5')
SiStripMonitorTrack.TrackLabel          = cms.string('')
SiStripMonitorTrack.OutputMEsInRootFile = cms.bool(True)
SiStripMonitorTrack.OutputFileName      = cms.string('testReal.root')
SiStripMonitorTrack.Cluster_src         = cms.string('siStripClusters')
SiStripMonitorTrack.Mod_On              = cms.bool(False)
SiStripMonitorTrack.OffHisto_On         = cms.bool(True)
SiStripMonitorTrack.FolderName          = cms.string('SiStrip/Tracks')

#-----------------------
#  Scheduling
#-----------------------
trackerGR = cms.Sequence(siStripDigis*offlineBeamSpot*striptrackerlocalreco*ctftracksP5)
DQMSiStripMonitorTrack_Real = cms.Sequence(trackerGR*SiStripMonitorTrack)


