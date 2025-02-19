import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_EventContent_cff import *
higgsToWW2LeptonsEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToWW2LeptonsEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
higgsToWW2LeptonsEventContentRECOSIM.outputCommands.extend(higgsToWW2LeptonsEventContent.outputCommands)

