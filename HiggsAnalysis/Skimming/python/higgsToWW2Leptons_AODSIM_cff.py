import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_EventContent_cff import *
higgsToWW2LeptonsEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToWW2LeptonsEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
higgsToWW2LeptonsEventContentAODSIM.outputCommands.extend(higgsToWW2LeptonsEventContent.outputCommands)

