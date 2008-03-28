import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToTauTau_ETau_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
AODSIMZToTauTauETauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
zToTauTau_ETauOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMZToTauTauETauEventContent,
    zToTauTauETauEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToTauTauETauAODSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('zToTauTauETau.root')
)

AODSIMZToTauTauETauEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMZToTauTauETauEventContent.outputCommands.extend(zToTauTauETauEventContent.outputCommands)

