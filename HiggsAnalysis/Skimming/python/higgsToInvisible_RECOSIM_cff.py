import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToInvisible_EventContent_cff import *
higgsToInvisibleEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToInvisibleEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
higgsToInvisibleEventContentRECOSIM.outputCommands.extend(higgsToInvisibleEventContent.outputCommands)

