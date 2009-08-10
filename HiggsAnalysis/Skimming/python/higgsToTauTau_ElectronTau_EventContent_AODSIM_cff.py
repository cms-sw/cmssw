import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_EventContent_cff import *
higgsToTauTauElectronTauEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToTauTauElectronTauEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
higgsToTauTauElectronTauEventContentAODSIM.outputCommands.extend(higgsToTauTauElectronTauEventContentAODSIM.outputCommands)

