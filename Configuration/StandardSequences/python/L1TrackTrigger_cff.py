import FWCore.ParameterSet.Config as cms

### put L1 track trigger configs here

from L1Trigger.TrackTrigger.TrackTrigger_cff import *


L1TrackTrigger=cms.Sequence(TrackTriggerClustersStubs)
