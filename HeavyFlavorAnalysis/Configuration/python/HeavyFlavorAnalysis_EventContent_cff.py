import FWCore.ParameterSet.Config as cms

#
# HeavyFlavorAnalysis event content 
#
from HeavyFlavorAnalysis.Skimming.bToMuMu_EventContent_cff import *
from HeavyFlavorAnalysis.Skimming.jpsiToMuMu_EventContent_cff import *
from HeavyFlavorAnalysis.Skimming.upsilonToMuMu_EventContent_cff import *
from HeavyFlavorAnalysis.Skimming.tauTo3Mu_EventContent_cff import *

HeavyFlavorAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
HeavyFlavorAnalysisEventContent.outputCommands.extend(bToMuMuEventContent.outputCommands)
HeavyFlavorAnalysisEventContent.outputCommands.extend(jpsiToMuMuEventContent.outputCommands)
HeavyFlavorAnalysisEventContent.outputCommands.extend(upsilonToMuMuEventContent.outputCommands)
HeavyFlavorAnalysisEventContent.outputCommands.extend(tauTo3MuEventContent.outputCommands)

