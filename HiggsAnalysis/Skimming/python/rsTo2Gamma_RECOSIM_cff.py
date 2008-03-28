import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from HiggsAnalysis.Skimming.rsTo2Gamma_EventContent_cff import *
rsTo2GammaEventContentRECOSIM = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
rsTo2GammaEventContentRECOSIM.outputCommands.extend(RECOSIMEventContent.outputCommands)
rsTo2GammaEventContentRECOSIM.outputCommands.extend(rsTo2GammaEventContent.outputCommands)

