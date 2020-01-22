import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from L1Trigger.TrackFindingTracklet.Tracklet_cfi import *

from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *
TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag(cms.InputTag("TTTracksFromTrackletEmulation", "Level1TTTracks") )

L1TrackletEmulationTracks = cms.Sequence(offlineBeamSpot*TTTracksFromTrackletEmulation)
L1TrackletEmulationTracksWithAssociators = cms.Sequence(offlineBeamSpot*TTTracksFromTrackletEmulation*TrackTriggerAssociatorTracks)

