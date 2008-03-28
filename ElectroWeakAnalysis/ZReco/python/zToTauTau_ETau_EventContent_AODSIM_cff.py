import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from ElectroWeakAnalysis.ZReco.zToTauTau_ETau_EventContent_cff import *
AODSIMZToTauTauETauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMZToTauTauETauEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMZToTauTauETauEventContent.outputCommands.extend(zToTauTauETauEventContent.outputCommands)

