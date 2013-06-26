import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.metHigh_EventContent_cff import *
AODSIMMetHighEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMMetHighEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMMetHighEventContent.outputCommands.extend(metHighEventContent.outputCommands)

