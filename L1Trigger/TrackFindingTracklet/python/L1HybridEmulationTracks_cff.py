import FWCore.ParameterSet.Config as cms

from RecoVertex.BeamSpotProducer.BeamSpot_cfi import *

from L1Trigger.TrackFindingTracklet.l1tTTTracksFromTrackletEmulation_cfi import *

from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *

from L1Trigger.TrackFindingTracklet.ProducerHPH_cff import *

# prompt hybrid emulation
TTTrackAssociatorFromPixelDigis.TTTracks = cms.VInputTag(cms.InputTag("l1tTTTracksFromTrackletEmulation", "Level1TTTracks") )

L1THybridTracks = cms.Sequence(offlineBeamSpot*l1tTTTracksFromTrackletEmulation)
L1THybridTracksWithAssociators = cms.Sequence(offlineBeamSpot*l1tTTTracksFromTrackletEmulation*TrackTriggerAssociatorTracks)

# extended hybrid (=displaced tracking) emulation
TTTrackAssociatorFromPixelDigisExtended = TTTrackAssociatorFromPixelDigis.clone(
    TTTracks = cms.VInputTag(cms.InputTag("l1tTTTracksFromExtendedTrackletEmulation", "Level1TTTracks") )
)

L1TExtendedHybridTracks = cms.Sequence(offlineBeamSpot*l1tTTTracksFromExtendedTrackletEmulation)
L1TExtendedHybridTracksWithAssociators = cms.Sequence(offlineBeamSpot*l1tTTTracksFromExtendedTrackletEmulation*TTTrackAssociatorFromPixelDigisExtended)

# both (prompt + extended) hybrid emulation 
L1TPromptExtendedHybridTracks = cms.Sequence(offlineBeamSpot*l1tTTTracksFromTrackletEmulation*l1tTTTracksFromExtendedTrackletEmulation)
L1TPromptExtendedHybridTracksWithAssociators = cms.Sequence(offlineBeamSpot*l1tTTTracksFromTrackletEmulation*TrackTriggerAssociatorTracks*l1tTTTracksFromExtendedTrackletEmulation*TTTrackAssociatorFromPixelDigisExtended)
