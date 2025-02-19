import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToInvisible_EventContent_cff import *
higgsToInvisibleEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToInvisibleEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
higgsToInvisibleEventContentAODSIM.outputCommands.extend(higgsToInvisibleEventContent.outputCommands)

