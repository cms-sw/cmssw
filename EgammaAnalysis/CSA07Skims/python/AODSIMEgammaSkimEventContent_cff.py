import FWCore.ParameterSet.Config as cms

# Egamma AODSIM event content 
# Saves standard AODSIM and Egamma-specific data (which might not be empty)
#
from Configuration.EventContent.EventContent_cff import *
from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
AODSIMEgammaSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMEgammaSkimEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMEgammaSkimEventContent.outputCommands.extend(egammaSkimEventContent.outputCommands)

