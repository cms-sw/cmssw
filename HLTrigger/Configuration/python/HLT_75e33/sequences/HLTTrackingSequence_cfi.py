import FWCore.ParameterSet.Config as cms

from ..modules.hltGeneralTracks_cfi import *
from ..modules.hltTrackerClusterCheck_cfi import *
from ..sequences.HLTHighPtTripletStepSequence_cfi import *
from ..sequences.HLTInitialStepSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *
from ..sequences.HLTPhase2PixelTracksAndVerticesSequence_cfi import *

_HLTTrackingSequenceLegacy = cms.Sequence(
    HLTItLocalRecoSequence
    +HLTOtLocalRecoSequence
    +hltTrackerClusterCheck
    +HLTPhase2PixelTracksAndVerticesSequence
    +HLTInitialStepSequence
    +HLTHighPtTripletStepSequence
    +hltGeneralTracks
)

HLTTrackingSequence = _HLTTrackingSequenceLegacy.copyAndExclude([HLTHighPtTripletStepSequence])

# Serial sequence for CPU vs. GPU validation, to be kept in sync with default sequence
HLTTrackingSequenceSerialSync = cms.Sequence(
    HLTItLocalRecoSequence
    +HLTOtLocalRecoSequence
    +hltTrackerClusterCheck
    +HLTPhase2PixelTracksAndVerticesSequenceSerialSync
    +HLTInitialStepSequenceSerialSync
    +hltGeneralTracks
)

from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
(ngtScouting & ~trackingLST).toReplaceWith(HLTTrackingSequence, HLTTrackingSequence.copyAndExclude([HLTInitialStepSequence]))

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking
hltPhase2LegacyTracking.toReplaceWith(HLTTrackingSequence, _HLTTrackingSequenceLegacy)
