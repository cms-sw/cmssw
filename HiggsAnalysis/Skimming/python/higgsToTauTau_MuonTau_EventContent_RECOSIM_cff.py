import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_MuonTau_EventContent_cff import *
higgsToTauTauMuonTauEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToTauTauMuonTauEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
higgsToTauTauMuonTauEventContentRECOSIM.outputCommands.extend(higgsToTauTauMuonTauEventContentRECOSIM.outputCommands)

