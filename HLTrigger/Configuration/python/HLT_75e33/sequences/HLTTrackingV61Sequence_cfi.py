import FWCore.ParameterSet.Config as cms

from ..modules.hltGeneralTracks_cfi import *
from ..modules.hltPhase2PixelVertices_cfi import *
from ..modules.hltPhase2TrimmedPixelVertices_cfi import *
from ..modules.hltTrackerClusterCheck_cfi import *
from ..sequences.HLTHighPtTripletStepSequence_cfi import *
from ..sequences.HLTPhase2PixelTracksSequence_cfi import *
from ..sequences.HLTInitialStepSequence_cfi import *
from ..sequences.HLTItLocalRecoSequence_cfi import *
from ..sequences.HLTOtLocalRecoSequence_cfi import *

HLTTrackingV61Sequence = cms.Sequence((HLTItLocalRecoSequence+HLTOtLocalRecoSequence+hltTrackerClusterCheck+HLTPhase2PixelTracksSequence+hltPhase2PixelVertices+HLTInitialStepSequence+HLTHighPtTripletStepSequence+hltGeneralTracks))

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(HLTTrackingV61Sequence, HLTTrackingV61Sequence.copyAndExclude([HLTHighPtTripletStepSequence]))

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
_HLTTrackingV61SequenceTrimming = HLTTrackingV61Sequence.copy()
_HLTTrackingV61SequenceTrimming.insert(_HLTTrackingV61SequenceTrimming.index(hltPhase2PixelVertices)+1, hltPhase2TrimmedPixelVertices)
phase2_hlt_vertexTrimming.toReplaceWith(HLTTrackingV61Sequence, _HLTTrackingV61SequenceTrimming)
