import FWCore.ParameterSet.Config as cms

# SiStrip DQM Reco

from DQM.SiStripMonitorClient.RecoForDQM_cff import *

# Refitting needed due to missing trajectories in RECO
from RecoTracker.TrackProducer.RefitterWithMaterial_cfi import *
# Clone for Cosmic Track Finder
import RecoTracker.TrackProducer.RefitterWithMaterial_cfi
cosmictrackfinderP5Refitter                   = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
cosmictrackfinderP5Refitter.src               = 'cosmictrackfinderP5'
cosmictrackfinderP5Refitter.TrajectoryInEvent = True
# Clone for CTF Track Finder
import RecoTracker.TrackProducer.RefitterWithMaterial_cfi
ctfWithMaterialTracksP5Refitter                   = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
ctfWithMaterialTracksP5Refitter.src               = 'ctfWithMaterialTracksP5'
ctfWithMaterialTracksP5Refitter.TrajectoryInEvent = True
# Clone for RS Track Finder
import RecoTracker.TrackProducer.RefitterWithMaterial_cfi
rsWithMaterialTracksP5Refitter                   = RecoTracker.TrackProducer.RefitterWithMaterial_cfi.TrackRefitter.clone()
rsWithMaterialTracksP5Refitter.src               = 'rsWithMaterialTracksP5'
rsWithMaterialTracksP5Refitter.TrajectoryInEvent = True

# Scheduling

SiStripDQMRecoFromRaw = cms.Sequence(
    siStripDigis     *
    siPixelDigis     *
    offlineBeamSpot  *
    trackerlocalreco *
    tracksP5
)

SiStripDQMRecoGlobalRunCAF = cms.Sequence(
    cosmictrackfinderP5Refitter     +
    ctfWithMaterialTracksP5Refitter +
    rsWithMaterialTracksP5Refitter
)
