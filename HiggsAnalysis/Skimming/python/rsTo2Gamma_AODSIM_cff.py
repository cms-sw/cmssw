import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.rsTo2Gamma_EventContent_cff import *
rsTo2GammaEventContentAODSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
rsTo2GammaEventContentAODSIM.outputCommands.extend(AODSIMEventContent.outputCommands)
rsTo2GammaEventContentAODSIM.outputCommands.extend(rsTo2GammaEventContent.outputCommands)

