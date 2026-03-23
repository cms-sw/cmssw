import FWCore.ParameterSet.Config as cms
from HeterogeneousCore.AlpakaCore.functions import makeSerialClone

from ..modules.hltPhase2PixelVertices_cfi import *

HLTPhase2PixelVertexingSequence = cms.Sequence(
    hltPhase2PixelVertices
)

# Empty sequence as a placeholder to be filled when alpakaValidationHLT is active
HLTPhase2PixelVertexingSequenceSerialSync = cms.Sequence()

hltPhase2PixelVerticesSerialSync = hltPhase2PixelVertices.clone(
    TrackCollection = "hltPhase2PixelTracksCAExtensionSerialSync"
)
# Serial sequence for CPU vs. GPU validation, to be kept in sync with default sequence
from Configuration.ProcessModifiers.alpakaValidationHLT_cff import alpakaValidationHLT
alpakaValidationHLT.toReplaceWith(HLTPhase2PixelVertexingSequenceSerialSync,
    cms.Sequence(hltPhase2PixelVerticesSerialSync)
)
