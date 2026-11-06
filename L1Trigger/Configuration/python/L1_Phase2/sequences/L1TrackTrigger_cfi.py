import FWCore.ParameterSet.Config as cms

from ..sequences.L1TPromptExtendedHybridTracksWithAssociators_cfi import *
from ..modules.ProducerDTC_cfi import *
from ..sequences.TrackTriggerAssociatorClustersStubs_cfi import *
from ..sequences.TrackTriggerClustersStubs_cfi import *

L1TrackTrigger = cms.Sequence((TrackTriggerClustersStubs+TrackTriggerAssociatorClustersStubs+ProducerDTC)+L1TPromptExtendedHybridTracksWithAssociators)
