import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.metHigh_EventContent_cff import *
FEVTSIMMetHighEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
FEVTSIMMetHighEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMMetHighEventContent.outputCommands.extend(metHighEventContent.outputCommands)

