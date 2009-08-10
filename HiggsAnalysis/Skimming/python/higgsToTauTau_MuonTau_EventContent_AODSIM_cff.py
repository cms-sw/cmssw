import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_EventContent_cff import *
higgsToTauTauMuonTauEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToTauTauMuonTauEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
higgsToTauTauMuonTauEventContentAODSIM.outputCommands.extend(higgsToTauTauMuonTauEventContentAODSIM.outputCommands)

