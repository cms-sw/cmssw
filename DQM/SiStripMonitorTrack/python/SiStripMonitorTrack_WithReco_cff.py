import FWCore.ParameterSet.Config as cms

#-----------------------
#  Reconstruction Modules
#-----------------------
# Real data raw to digi
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'source'

# SiPixel data raw to digi
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
siPixelDigis.InputLabel = 'source'

# Local and Track Reconstruction
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

#-----------------------
#  SiStripMonitorTrack
#-----------------------
from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack.TrackProducer       = cms.string('ctfWithMaterialTracksP5')
SiStripMonitorTrack.TrackLabel          = cms.string('')
SiStripMonitorTrack.Cluster_src         = cms.string('siStripClusters')
SiStripMonitorTrack.Mod_On              = cms.bool(False)
SiStripMonitorTrack.OffHisto_On         = cms.bool(True)
SiStripMonitorTrack.HistoFlag_On        = cms.bool(False)
SiStripMonitorTrack.FolderName          = cms.string('SiStrip/Tracks')

#-----------------------
#  Scheduling
#-----------------------
trackerGR = cms.Sequence(siPixelDigis*siStripDigis*offlineBeamSpot*trackerlocalreco*ctftracksP5)
DQMSiStripMonitorTrack_Real = cms.Sequence(trackerGR*SiStripMonitorTrack)
