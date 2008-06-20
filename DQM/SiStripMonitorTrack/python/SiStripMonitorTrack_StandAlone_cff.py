import FWCore.ParameterSet.Config as cms

from DQM.SiStripMonitorTrack.SiStripMonitorTrack_cfi import *
#TrackRefitter With Material
from RecoTracker.TrackProducer.RefitterWithMaterial_cff import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *
#-----------------------
#  Reconstruction Modules
#-----------------------
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
DQMSiStripMonitorTrack_Sim = cms.Sequence(siStripDigis*siStripZeroSuppression*TrackRefitter*SiStripMonitorTrack)

