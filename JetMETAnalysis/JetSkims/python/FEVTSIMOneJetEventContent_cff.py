import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.JetSkims.onejet_EventContent_cff import *
FEVTSIMOneJetEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
FEVTSIMOneJetEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMOneJetEventContent.outputCommands.extend(onejetEventContent.outputCommands)

