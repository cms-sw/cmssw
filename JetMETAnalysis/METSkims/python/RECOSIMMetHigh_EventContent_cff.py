import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.metHigh_EventContent_cff import *
RECOSIMMetHighEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMMetHighEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMMetHighEventContent.outputCommands.extend(metHighEventContent.outputCommands)

