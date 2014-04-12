import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.sumET_EventContent_cff import *
AODSIMSumETEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
AODSIMSumETEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMSumETEventContent.outputCommands.extend(sumETEventContent.outputCommands)

