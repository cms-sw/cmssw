import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelVertices_cfi import *

HLTPhase2PixelVertexingSequence = cms.Sequence(
    hltPhase2PixelVertices
)

from Configuration.ProcessModifiers.phase2_hlt_vertexTrimming_cff import phase2_hlt_vertexTrimming
from ..modules.hltPhase2TrimmedPixelVertices_cfi import *
_HLTPhase2PixelVertexingSequenceTrimming = HLTPhase2PixelVertexingSequence.copy()
_HLTPhase2PixelVertexingSequenceTrimming.insert(
    _HLTPhase2PixelVertexingSequenceTrimming.index(hltPhase2PixelVertices)+1, 
    hltPhase2TrimmedPixelVertices
    )
phase2_hlt_vertexTrimming.toReplaceWith(HLTPhase2PixelVertexingSequence, _HLTPhase2PixelVertexingSequenceTrimming)
