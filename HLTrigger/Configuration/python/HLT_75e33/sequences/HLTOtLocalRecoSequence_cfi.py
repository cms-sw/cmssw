import FWCore.ParameterSet.Config as cms

from ..modules.hltMeasurementTrackerEvent_cfi import *
from ..modules.hltSiPhase2RecHits_cfi import *

HLTOtLocalRecoSequence = cms.Sequence(hltMeasurementTrackerEvent
                                      +hltSiPhase2RecHits
                                      )

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
hltPhase2LegacyTracking.toReplaceWith(HLTOtLocalRecoSequence,
                                      HLTOtLocalRecoSequence.copyAndExclude([hltSiPhase2RecHits])
                                      )

_HLTOtLocalRecoSequenceWithHits = cms.Sequence(hltMeasurementTrackerEvent
                                               +hltSiPhase2RecHits
                                               )

# Restore the OT rechits that hltPhase2LegacyTracking removes: the CA extension and the vector hit chain need them.
from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
from Configuration.ProcessModifiers.phase2CAVectorHits_cff import phase2CAVectorHits
(phase2CAExtension | phase2CAVectorHits).toReplaceWith(HLTOtLocalRecoSequence, _HLTOtLocalRecoSequenceWithHits)
