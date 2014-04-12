import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.metLow_EventContent_cff import *
AODSIMMetLowEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMMetLowEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMMetLowEventContent.outputCommands.extend(metLowEventContent.outputCommands)

