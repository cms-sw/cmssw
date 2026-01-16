import FWCore.ParameterSet.Config as cms

from ..modules.hltPhase2PixelVertices_cfi import *

HLTPhase2PixelVertexingSequence = cms.Sequence(
    hltPhase2PixelVertices
)
