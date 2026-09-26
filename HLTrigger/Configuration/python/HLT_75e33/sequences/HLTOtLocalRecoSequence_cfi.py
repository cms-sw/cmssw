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

# Restore the OT rechits that hltPhase2LegacyTracking removes: the CA extension and the stub chain need them.
from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension
from Configuration.ProcessModifiers.phase2CAStubs_cff import phase2CAStubs
from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
(phase2CAExtension | phase2CAStubs | ngtScouting).toReplaceWith(HLTOtLocalRecoSequence, _HLTOtLocalRecoSequenceWithHits)
