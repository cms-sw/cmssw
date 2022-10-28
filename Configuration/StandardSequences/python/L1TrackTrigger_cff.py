import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.TrackTrigger_cff import *
from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *
from L1Trigger.TrackerDTC.ProducerED_cff import *
from L1Trigger.TrackFindingTracklet.L1HybridEmulationTracks_cff import * 

L1TrackTrigger=cms.Sequence(TrackTriggerClustersStubs*TrackTriggerAssociatorClustersStubs*TrackerDTCProducer)

# Customisation to enable TTTracks in geometry D41 and later (corresponding to phase2_trackerV14 or later). Includes the HGCAL L1 trigger
_tttracks_l1tracktrigger = L1TrackTrigger.copy()
_tttracks_l1tracktrigger = cms.Sequence(_tttracks_l1tracktrigger + L1TPromptExtendedHybridTracksWithAssociators)

from Configuration.Eras.Modifier_phase2_trigger_cff import phase2_trigger
phase2_trigger.toReplaceWith( L1TrackTrigger, _tttracks_l1tracktrigger )
