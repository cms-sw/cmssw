import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.hltPhase2LegacyTracking_cff import hltPhase2LegacyTracking

# This modifier reverts the Tracking in the HLT Phase-2 to the Legacy algorithms from Run-2
# but uses Patatrack quads for the initial step.
hltPhase2LegacyTrackingPatatrackQuads = cms.Modifier()
# The modifier is meant to be used only together with the hltPhase2LegacyTracking procModifier,
# so a modifier chain is defined to be for all practical purposes.
hltPhase2LegacyTrackingPatatrackQuadsChain = cms.ModifierChain(hltPhase2LegacyTracking, hltPhase2LegacyTrackingPatatrackQuads)
