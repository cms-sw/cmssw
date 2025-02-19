import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_EventContent_cff import *
heavyChHiggsToTauNuEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
heavyChHiggsToTauNuEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
heavyChHiggsToTauNuEventContentRECOSIM.outputCommands.extend(heavyChHiggsToTauNuEventContent.outputCommands)

