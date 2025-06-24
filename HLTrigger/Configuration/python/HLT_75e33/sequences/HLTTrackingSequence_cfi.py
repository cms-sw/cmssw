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
from ..modules.hltPhase2PixelTracksCutClassifier_cfi import hltPhase2PixelTracksCutClassifier
from ..modules.hltPhase2PixelTracksSelectionHighPurity_cfi import hltPhase2PixelTracksSelectionHighPurity

HLTTrackingSequence = cms.Sequence(HLTItLocalRecoSequence+
                                   HLTOtLocalRecoSequence+
                                   hltTrackerClusterCheck+
                                   HLTPhase2PixelTracksSequence+
                                   hltPhase2PixelVertices+
                                   hltPhase2PixelTracksCutClassifier+
                                   hltPhase2PixelTracksSelectionHighPurity+
                                   HLTInitialStepSequence+
                                   HLTHighPtTripletStepSequence+
                                   hltGeneralTracks)

from Configuration.ProcessModifiers.singleIterPatatrack_cff import singleIterPatatrack
singleIterPatatrack.toReplaceWith(HLTTrackingSequence, HLTTrackingSequence.copyAndExclude([HLTHighPtTripletStepSequence]))

from Configuration.ProcessModifiers.ngtScouting_cff import ngtScouting
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
(ngtScouting & trackingLST).toReplaceWith(HLTTrackingSequence, HLTTrackingSequence.copyAndExclude([HLTHighPtTripletStepSequence]))
(ngtScouting & ~trackingLST).toReplaceWith(HLTTrackingSequence, HLTTrackingSequence.copyAndExclude([HLTInitialStepSequence,HLTHighPtTripletStepSequence]))

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
_HLTTrackingSequenceTrimming = HLTTrackingSequence.copy()
_HLTTrackingSequenceTrimming.insert(_HLTTrackingSequenceTrimming.index(hltPhase2PixelVertices)+1, hltPhase2TrimmedPixelVertices)
phase2_hlt_vertexTrimming.toReplaceWith(HLTTrackingSequence, _HLTTrackingSequenceTrimming)
