import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_cff import *
higgsToTauTauLeptonTauEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToTauTauLeptonTauEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
higgsToTauTauLeptonTauEventContentAODSIM.outputCommands.extend(higgsToTauTauLeptonTauEventContentAODSIM.outputCommands)

