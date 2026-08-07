import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.alpaka_cff import alpaka
from Configuration.ProcessModifiers.pixelTrackMask_cff import pixelTrackMask
from Configuration.ProcessModifiers.phase2CAExtension_cff import phase2CAExtension

# collect all PixelTrackMask-related process modifiers here
allPixelTrackMask = cms.ModifierChain(alpaka,pixelTrackMask,phase2CAExtension)
