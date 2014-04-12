import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_EventContent_cff import *
higgsTo2GammaEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsTo2GammaEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
higgsTo2GammaEventContentAODSIM.outputCommands.extend(higgsTo2GammaEventContent.outputCommands)

