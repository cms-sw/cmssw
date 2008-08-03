import FWCore.ParameterSet.Config as cms

# SiStrip DQM Reco
# Digitiser ####
from EventFilter.SiStripRawToDigi.SiStripDigis_cfi import *
siStripDigis.ProductLabel = 'source'
from EventFilter.SiPixelRawToDigi.SiPixelRawToDigi_cfi import *
siPixelDigis.InputLabel = 'source'

# Local Reco ####
from RecoLocalTracker.Configuration.RecoLocalTracker_Cosmics_cff import *
siStripClusters.SiStripQualityLabel = ''

# Track Reconstruction ########
from RecoTracker.Configuration.RecoTrackerP5_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
CTF_P5_MeasurementTracker.pixelClusterProducer = ''
RS_P5_MeasurementTracker.pixelClusterProducer = ''

# Refitting needed due to missing trajectories in RECO
from RecoTracker.TrackProducer.RefitterWithMaterial_cfi import *
RKFittingSmoother.ComponentName = 'FittingSmootherRK'
# Clone for Cosmic Track Finder
import RecoTracker.TrackProducer.RefitterWithMaterial_cfi
cosmictrackfinderP5Refitter = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
cosmictrackfinderP5Refitter.src = 'cosmictrackfinderP5'
cosmictrackfinderP5Refitter.TrajectoryInEvent = True
# Clone for CTF Track Finder
import RecoTracker.TrackProducer.RefitterWithMaterial_cfi
ctfWithMaterialTracksP5Refitter = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
ctfWithMaterialTracksP5Refitter.src = 'ctfWithMaterialTracksP5'
ctfWithMaterialTracksP5Refitter.TrajectoryInEvent = True
# Clone for RS Track Finder
import RecoTracker.TrackProducer.RefitterWithMaterial_cfi
rsWithMaterialTracksP5Refitter = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
rsWithMaterialTracksP5Refitter.src = 'rsWithMaterialTracksP5'
rsWithMaterialTracksP5Refitter.TrajectoryInEvent = True

# Scheduling
SiStripDQMRecoGlobalRunCAF = cms.Sequence(cosmictrackfinderP5Refitter+ctfWithMaterialTracksP5Refitter+rsWithMaterialTracksP5Refitter)



