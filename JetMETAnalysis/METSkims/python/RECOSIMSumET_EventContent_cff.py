import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from JetMETAnalysis.METSkims.sumET_EventContent_cff import *
RECOSIMSumETEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
RECOSIMSumETEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMSumETEventContent.outputCommands.extend(sumETEventContent.outputCommands)

