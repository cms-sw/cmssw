import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
from ElectroWeakAnalysis.ZReco.zToEE_EventContent_cff import *
AODSIMZToEEEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
zToEEOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMZToEEEventContent,
    zToEEEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToEE'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('zToEE.root')
)

AODSIMZToEEEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMZToEEEventContent.outputCommands.extend(zToEEEventContent.outputCommands)

