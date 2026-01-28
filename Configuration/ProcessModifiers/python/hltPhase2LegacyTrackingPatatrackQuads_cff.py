import FWCore.ParameterSet.Config as cms

# This modifiers reverts the Tracking in the HLT Phase-2 to the Legacy algorithms from Run-2
# but uses Patatrack quads for the initial step.
# It is meant to be used only together with the hltPhase2LegacyTracking procModifier.
hltPhase2LegacyTrackingPatatrackQuads = cms.Modifier()
