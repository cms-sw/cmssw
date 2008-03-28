import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_EventContent_cff import *
heavyChHiggsToTauNuEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
heavyChHiggsToTauNuEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
heavyChHiggsToTauNuEventContentAODSIM.outputCommands.extend(heavyChHiggsToTauNuEventContent.outputCommands)

