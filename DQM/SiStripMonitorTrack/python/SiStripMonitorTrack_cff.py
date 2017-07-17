import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
# TrackInfo ####
from RecoTracker.TrackProducer.TrackRefitters_cff import *
#-----------------------
#  Reconstruction Modules
#-----------------------
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
DQMSiStripMonitorTrack_Sim = cms.Sequence(siStripDigis*siStripZeroSuppression*TrackRefitter*SiStripMonitorTrack)
DQMSiStripMonitorTrack_Real = cms.Sequence(SiStripMonitorTrack)
SiStripMonitorTrack.TrackProducer = 'TrackRefitter'
SiStripMonitorTrack.TrackLabel = ''
SiStripMonitorTrack.Cluster_src = 'siStripClusters'
SiStripMonitorTrack.Mod_On = False
SiStripMonitorTrack.OffHisto_On = False
TrackRefitter.TrajectoryInEvent = True

