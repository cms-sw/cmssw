import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HeavyFlavorAnalysis.Skimming.tauTo3Mu_EventContent_cff import *
AODSIMTauTo3MuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMTauTo3MuEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMTauTo3MuEventContent.outputCommands.extend(tauTo3MuEventContent.outputCommands)

