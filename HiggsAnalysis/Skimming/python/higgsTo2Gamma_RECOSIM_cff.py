import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.higgsTo2Gamma_EventContent_cff import *
higgsTo2GammaEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
higgsTo2GammaEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
higgsTo2GammaEventContentRECOSIM.outputCommands.extend(higgsTo2GammaEventContent.outputCommands)

