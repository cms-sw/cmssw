import FWCore.ParameterSet.Config as cms

#-------------------------
#  Reconstruction Modules
#-------------------------
# Digitizer ####
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *

# Local Reco ####
from RecoLocalTracker.SiStripZeroSuppression.SiStripZeroSuppression_cfi import *
from RecoLocalTracker.SiPixelRecHits.PixelCPEESProducers_cff import *

# TrackRefitter With Material
from RecoTracker.TrackProducer.TrackRefitters_cff import *
TrackRefitter.TrajectoryInEvent = True

# Sequence
RecoModulesForSimData = cms.Sequence(siStripDigis*siStripZeroSuppression*TrackRefitter)


