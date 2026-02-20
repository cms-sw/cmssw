import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

from ..modules.hltPhase2PixelVertices_cfi import *

HLTPhase2PixelVertexingSequence = cms.Sequence(
    hltPhase2PixelVertices
)

# Serial sequence for CPU vs. GPU validation, to be kept in sync with default sequence
hltPhase2PixelVerticesSerialSync = hltPhase2PixelVertices.clone(
    TrackCollection = "hltPhase2PixelTracksCAExtensionSerialSync"
)
HLTPhase2PixelVertexingSequenceSerialSync = cms.Sequence(
    hltPhase2PixelVerticesSerialSync
)
