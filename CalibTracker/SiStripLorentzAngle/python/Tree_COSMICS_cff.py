import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripLorentzAngle.Tree_RECO_cff import *

trackInputTag = cms.InputTag("ctfWithMaterialTracksP5","")
shallowTracks.Tracks = trackInputTag
shallowTrackClusters.Tracks = trackInputTag

oneplustracks = cms.EDFilter( "TrackCountFilter", src = trackInputTag, minNumber = cms.uint32(1) )

#tracking
from RecoTracker.Configuration.RecoTrackerP5_cff import *
ctfWithMaterialTracksP5.TrajectoryInEvent = True
recotrackCosmics = cms.Sequence( offlineBeamSpot + siPixelRecHits*siStripMatchedRecHits*ctftracksP5)

#Schedule
everything_step = cms.Path( oneplustracks+recotrackCosmics+theBigNtuple )
schedule = cms.Schedule( everything_step )
