import FWCore.ParameterSet.Config as cms

from L1Trigger.TrackTrigger.TrackTrigger_cff import *
##from SimTracker.TrackTriggerAssociation.TrackTriggerAssociator_cff import *

#L1TrackTrigger=cms.Sequence(TrackTriggerClustersStubs*TrackTriggerAssociatorClustersStubs*TrackTriggerTTTracks*TrackTriggerAssociatorTracks)
L1TrackTrigger=cms.Sequence(TrackTriggerClustersStubs)
