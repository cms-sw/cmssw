
import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.lightChHiggsToTauNu_EventContent_cff import *
lightChHiggsToTauNuEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
lightChHiggsToTauNuEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
lightChHiggsToTauNuEventContentAODSIM.outputCommands.extend(lightChHiggsToTauNuEventContent.outputCommands)




