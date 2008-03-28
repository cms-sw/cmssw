import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.metLow_EventContent_cff import *
FEVTSIMMetLowEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
FEVTSIMMetLowEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMMetLowEventContent.outputCommands.extend(metLowEventContent.outputCommands)

