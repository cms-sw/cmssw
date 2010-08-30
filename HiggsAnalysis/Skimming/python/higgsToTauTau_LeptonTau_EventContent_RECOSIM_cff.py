import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_cff import *
higgsToTauTauLeptonTauEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToTauTauLeptonTauEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
higgsToTauTauLeptonTauEventContentRECOSIM.outputCommands.extend(higgsToTauTauLeptonTauEventContentRECOSIM.outputCommands)

