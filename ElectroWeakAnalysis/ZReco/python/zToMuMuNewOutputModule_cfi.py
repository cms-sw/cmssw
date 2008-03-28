import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuNew_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
AODSIMZToMuMuNewEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
zToMuMuNewOutputModule = cms.OutputModule("PoolOutputModule",
    zToMuMuNewEventSelection,
    AODSIMZToMuMuNewEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToMuMuNew'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('zToMuMuNew.root')
)

AODSIMZToMuMuNewEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMZToMuMuNewEventContent.outputCommands.extend(zToMuMuNewEventContent.outputCommands)

