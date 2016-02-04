import FWCore.ParameterSet.Config as cms

#
# HiggsAnalysis event content 
#
# Dominique Fortin - UC Riverside
from HiggsAnalysis.Skimming.heavyChHiggsToTauNu_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToInvisible_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToTauTau_LeptonTau_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToWW2Leptons_EventContent_cff import *
from HiggsAnalysis.Skimming.higgsToZZ4Leptons_EventContent_cff import *
# include "HiggsAnalysis/Skimming/data/rsTo2Gamma_EventContent.cff"
HiggsAnalysisEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
HiggsAnalysisEventContent.outputCommands.extend(heavyChHiggsToTauNuEventContent.outputCommands)
HiggsAnalysisEventContent.outputCommands.extend(higgsTo2GammaEventContent.outputCommands)
HiggsAnalysisEventContent.outputCommands.extend(higgsToInvisibleEventContent.outputCommands)
HiggsAnalysisEventContent.outputCommands.extend(higgsToTauTauLeptonTauEventContent.outputCommands)
HiggsAnalysisEventContent.outputCommands.extend(higgsToWW2LeptonsEventContent.outputCommands)
HiggsAnalysisEventContent.outputCommands.extend(higgsToZZ4LeptonsEventContent.outputCommands)

