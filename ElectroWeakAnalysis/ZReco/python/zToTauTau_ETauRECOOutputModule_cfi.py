import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToTauTau_ETau_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
RECOSIMZToTauTauETauEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
zToTauTau_ETauRECOOutputModule = cms.OutputModule("PoolOutputModule",
    RECOSIMZToTauTauETauEventContent,
    zToTauTauETauEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToTauTauETauRECOSIM'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('zToTauTauETauRECO.root')
)

RECOSIMZToTauTauETauEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMZToTauTauETauEventContent.outputCommands.extend(zToTauTauETauEventContent.outputCommands)

