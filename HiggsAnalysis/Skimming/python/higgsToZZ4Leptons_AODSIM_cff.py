import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_EventContent_cff import *
higgsToZZ4LeptonsEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToZZ4LeptonsEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
higgsToZZ4LeptonsEventContentAODSIM.outputCommands.extend(higgsToZZ4LeptonsEventContent.outputCommands)

