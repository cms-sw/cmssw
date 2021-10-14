import FWCore.ParameterSet.Config as cms

### SiStrip DQM Reconstruction ###

from DQM.SiStripMonitorClient.RecoForDQM_Cosmic_cff import *

## Refitting ##
from RecoTracker.TrackProducer.TrackRefitters_cff import *
# cosmic track finder #
import RecoTracker.TrackProducer.TrackRefitters_cff 
cosmictrackfinderP5Refitter = RecoTracker.TrackProducer.TrackRefitters_cff.TrackRefitter.clone(
    src = 'cosmictrackfinderP5',
    TrajectoryInEvent = True
)
# CTF #
import RecoTracker.TrackProducer.TrackRefitters_cff
ctfWithMaterialTracksP5Refitter = RecoTracker.TrackProducer.TrackRefitters_cff.TrackRefitter.clone(
    src = 'ctfWithMaterialTracksP5',
    TrajectoryInEvent = True
)
# RS #
import RecoTracker.TrackProducer.TrackRefitters_cff
rsWithMaterialTracksP5Refitter = RecoTracker.TrackProducer.TrackRefitters_cff.TrackRefitter.clone(
    src = 'rsWithMaterialTracksP5',
    TrajectoryInEvent = True
)

## Scheduling ##
# additional reco needed for running from RAW #
SiStripDQMRecoFromRaw = cms.Sequence(
    siStripDigis     *
    siPixelDigis     *
    offlineBeamSpot  *
    trackerlocalreco *
    tracksP5
)
# reco needed for DQM #
SiStripDQMRecoGlobalRunCAF = cms.Sequence(
    cosmictrackfinderP5Refitter     +
    ctfWithMaterialTracksP5Refitter +
    rsWithMaterialTracksP5Refitter
)
