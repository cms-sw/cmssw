import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_EventContent_cff import *
higgsToZZ4LeptonsEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsToZZ4LeptonsEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
higgsToZZ4LeptonsEventContentRECOSIM.outputCommands.extend(higgsToZZ4LeptonsEventContent.outputCommands)

