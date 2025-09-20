import FWCore.ParameterSet.Config as cms

from ..modules.hltMeasurementTrackerEvent_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *

HLTOtLocalRecoSequence = cms.Sequence(hltMeasurementTrackerEvent)

_HLTOtLocalRecoSequenceWithHits = cms.Sequence(hltMeasurementTrackerEvent
                                               +hltSiPhase2RecHits
                                               )

from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
phase2CAExtension.toReplaceWith(HLTOtLocalRecoSequence, _HLTOtLocalRecoSequenceWithHits)
