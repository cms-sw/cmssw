import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.lightChHiggsToTauNu_EventContent_cff import *
lightChHiggsToTauNuEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
lightChHiggsToTauNuEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
lightChHiggsToTauNuEventContentRECOSIM.outputCommands.extend(lightChHiggsToTauNuEventContent.outputCommands)

