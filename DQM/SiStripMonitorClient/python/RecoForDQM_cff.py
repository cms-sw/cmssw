

import FWCore.ParameterSet.Config as cms

# Digitiser ####
# SiStrip 
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'source'
# SiPixel
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
siPixelDigis.InputLabel = 'source'

# Local Reco ####    
from RecoLocalTracker.Configuration.RecoLocalTracker_cff import *
#DefaultClusterizer.ConditionsLabel = ''   #not needed to specify it is used as default

# Track Reconstruction ########
from RecoTracker.Configuration.RecoTracker_cff import *

# Beam Spot ########
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *

# Pixel Vertex
from RecoTracker.Configuration.RecoPixelVertexing_cff import *

# Reconstruction Sequence
RecoForDQMCollision = cms.Sequence(siPixelDigis*siStripDigis*trackerlocalreco*offlineBeamSpot*recopixelvertexing*ckftracks)



