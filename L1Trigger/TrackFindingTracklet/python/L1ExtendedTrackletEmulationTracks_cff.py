import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from L1Trigger.TrackFindingTracklet.Tracklet_cfi import *

from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *
TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag(cms.InputTag("TTTracksFromExtendedTrackletEmulation", "Level1TTTracks") )

L1ExtendedTrackletEmulationTracks = cms.Sequence(offlineBeamSpot*TTTracksFromExtendedTrackletEmulation)
L1ExtendedTrackletEmulationTracksWithAssociators = cms.Sequence(offlineBeamSpot*TTTracksFromExtendedTrackletEmulation*TrackTriggerAssociatorTracks)

