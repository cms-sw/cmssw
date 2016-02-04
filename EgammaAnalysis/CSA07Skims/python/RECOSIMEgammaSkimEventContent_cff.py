import FWCore.ParameterSet.Config as cms

# Egamma RECOSIM event content 
# Saves standard RECOSIM and Egamma-specific data (which might not be empty)
#
from Configuration.EventContent.EventContent_cff import *
from EgammaAnalysis.CSA07Skims.EgammaSkimEventContent_cff import *
RECOSIMEgammaSkimEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMEgammaSkimEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMEgammaSkimEventContent.outputCommands.extend(egammaSkimEventContent.outputCommands)

