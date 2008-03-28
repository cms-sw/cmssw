import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMu_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
AODSIMZToMuMuEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
zToMuMuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMZToMuMuEventContent,
    zToMuMuEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToMuMu'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('zToMuMu.root')
)

AODSIMZToMuMuEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMZToMuMuEventContent.outputCommands.extend(zToMuMuEventContent.outputCommands)

