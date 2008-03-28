import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuGolden_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
AODSIMZToMuMuGoldenEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
zToMuMuGoldenOutputModule = cms.OutputModule("PoolOutputModule",
    zToMuMuGoldenEventSelection,
    AODSIMZToMuMuGoldenEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToMuMuGoldenAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('zToMuMuGolden.root')
)

AODSIMZToMuMuGoldenEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMZToMuMuGoldenEventContent.outputCommands.extend(zToMuMuGoldenEventContent.outputCommands)

