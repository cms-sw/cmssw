import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.sumET_EventContent_cff import *
FEVTSIMSumETEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
FEVTSIMSumETEventContent.outputCommands.extend(FEVTSIMEventContent.outputCommands)
FEVTSIMSumETEventContent.outputCommands.extend(sumETEventContent.outputCommands)

