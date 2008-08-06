import FWCore.ParameterSet.Config as cms

# Digitiser ####
# SiStrip 
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'source'
# SiPixel
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
siPixelDigis.InputLabel = 'source'

# Local Reco ####    
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
siStripClusters.SiStripQualityLabel = ''

# Track Reconstruction ########
from RecoTracker.Configuration.RecoTrackerP5_cff import *
CTF_P5_MeasurementTracker.pixelClusterProducer = ''
RS_P5_MeasurementTracker.pixelClusterProducer = ''

# Beam Spot ########
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

# Reconstruction Sequence
RecoForDQM = cms.Sequence(siStripDigis*offlineBeamSpot*striptrackerlocalreco*ctftracksP5)



