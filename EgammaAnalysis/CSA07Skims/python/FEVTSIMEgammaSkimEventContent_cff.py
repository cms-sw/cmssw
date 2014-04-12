import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
FEVTSIMEgammaSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
FEVTSIMEgammaSkimEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMEgammaSkimEventContent.outputCommands.extend(egammaSkimEventContent.outputCommands)

