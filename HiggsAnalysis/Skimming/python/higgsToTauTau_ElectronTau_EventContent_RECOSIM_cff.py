import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_ElectronTau_EventContent_cff import *
higgsToTauTauElectronTauEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToTauTauElectronTauEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
higgsToTauTauElectronTauEventContentRECOSIM.outputCommands.extend(higgsToTauTauElectronTauEventContentRECOSIM.outputCommands)

