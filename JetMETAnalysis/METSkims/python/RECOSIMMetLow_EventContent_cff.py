import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.metLow_EventContent_cff import *
RECOSIMMetLowEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMMetLowEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMMetLowEventContent.outputCommands.extend(metLowEventContent.outputCommands)

