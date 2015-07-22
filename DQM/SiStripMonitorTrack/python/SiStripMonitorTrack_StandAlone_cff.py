import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
SiStripMonitorTrack.TrackProducer = 'TrackRefitter'
SiStripMonitorTrack.TrackLabel    = ''

SiStripMonitorTrack.Cluster_src = 'siStripClusters'
SiStripMonitorTrack.Mod_On        = True
SiStripMonitorTrack.OffHisto_On   = True
SiStripMonitorTrack.HistoFlag_On  = False
SiStripMonitorTrack.Trend_On      = False
#SiStripMonitorTrack.CCAnalysis_On = False

#TrackRefitter With Material
from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitter.src  = 'ctfWithMaterialTracksP5'
TrackRefitter.TrajectoryInEvent = True

#from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *

#-----------------------
#  Reconstruction Modules
#-----------------------
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *

DQMSiStripMonitorTrack_Sim = cms.Sequence( siStripDigis
                                           *
                                           siStripZeroSuppression
                                           *
                                           TrackRefitter
                                           *
                                           SiStripMonitorTrack
                                           )


# reconstruction sequence for Cosmics
from Configuration.StandardSequences.ReconstructionCosmics_cff import *

DQMSiStripMonitorTrack_CosmicSim = cms.Sequence( trackerCosmics
                                                 *
                                                 TrackRefitter
                                                 *
                                                 SiStripMonitorTrack
                                                 )

DQMSiStripMonitorTrack_Real = cms.Sequence(TrackRefitter
                                           *
                                           SiStripMonitorTrack
                                           )
