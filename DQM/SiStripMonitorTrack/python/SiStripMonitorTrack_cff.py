import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
# TrackInfo ####
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
#include "AnalysisAlgos/TrackInfoProducer/data/TrackInfoProducer.cfi"
#replace trackinfo.cosmicTracks=TrackRefitter
#replace trackinfo.rechits=TrackRefitter
#-----------------------
#  Reconstruction Modules
#-----------------------
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_SimData_cfi import *
DQMSiStripMonitorTrack_Sim = cms.Sequence(siStripZeroSuppression*TrackRefitter*SiStripMonitorTrack)
DQMSiStripMonitorTrack_Real = cms.Sequence(SiStripMonitorTrack)
SiStripMonitorTrack.TrackProducer = 'TrackRefitter'
SiStripMonitorTrack.TrackLabel = ''
#replace SiStripMonitorTrack.TrackInfo = "trackinfo"
SiStripMonitorTrack.OutputMEsInRootFile = True
SiStripMonitorTrack.OutputFileName = '/tmp/sistripmonitortrack_prova.root'
SiStripMonitorTrack.Cluster_src = 'siStripClusters'
SiStripMonitorTrack.Mod_On = False
TrackRefitter.TrajectoryInEvent = True

