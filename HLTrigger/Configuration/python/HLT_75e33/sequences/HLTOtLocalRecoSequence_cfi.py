import FWCore.ParameterSet.Config as cms

from ..modules.hltMeasurementTrackerEvent_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *
from RecoLocalTracker.Phase2TrackerRecHits.phase2OTRecHitsSoAConverter_cfi import *

from Configuration.ProcessModifiers.alpaka_cff import alpaka

HLTOtLocalRecoSequence = cms.Sequence(hltMeasurementTrackerEvent)

HLTOtLocalRecoSequenceWithHits_ = cms.Sequence(hltMeasurementTrackerEvent
                                               +hltSiPhase2RecHits
                                               +phase2OTRecHitsSoAConverter)

alpaka.toReplaceWith(HLTOtLocalRecoSequence, HLTOtLocalRecoSequenceWithHits_)
