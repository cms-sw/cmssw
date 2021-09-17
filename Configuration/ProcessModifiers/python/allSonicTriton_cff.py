import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.enableSonicTriton_cff import enableSonicTriton

# collect all SonicTriton-related process modifiers here
allSonicTriton = cms.ModifierChain(enableSonicTriton)
