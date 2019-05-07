import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from L1Trigger.TrackFindingTracklet.Tracklet_cfi import *

from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *
TTClusterAssociatorFromPixelDigis.digiSimLinks = cms.InputTag("simSiPixelDigis","Tracker")
TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag(cms.InputTag("TTTracksFromTracklet", "Level1TTTracks") )

L1TrackletTracks = cms.Sequence(offlineBeamSpot*TTTracksFromTracklet)
L1TrackletTracksWithAssociators = cms.Sequence(offlineBeamSpot*TTTracksFromTracklet*TrackTriggerAssociatorTracks)

