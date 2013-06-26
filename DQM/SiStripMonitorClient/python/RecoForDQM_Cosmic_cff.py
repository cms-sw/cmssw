

import FWCore.ParameterSet.Config as cms

# Digitiser ####
# SiStrip 
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'source'
# SiPixel
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
siPixelDigis.InputLabel = 'source'


# Local Reco Cosmic ####    
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
#DefaultClusterizer.QualityLabel = ''   #not needed to specify it is used as default

# Track Reconstruction Cosmic ########
from RecoTracker.Configuration.RecoTrackerP5_cff import *

# Beam Spot ########
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

# Reconstruction Sequence
RecoForDQMCosmic =  cms.Sequence(siPixelDigis*siStripDigis*offlineBeamSpot*trackerlocalreco*ctftracksP5)



