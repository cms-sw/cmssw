import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.zToMuMuRECO_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
RECOSIMZToMuMuRECOEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
zToMuMuRECOOutputModule = cms.OutputModule("PoolOutputModule",
    zToMuMuRECOEventSelection,
    RECOSIMZToMuMuRECOEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToMuMuRECO'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('zToMuMuRECO.root')
)

RECOSIMZToMuMuRECOEventContent.outputCommands.extend(RECOSIMEventContent.outputCommands)
RECOSIMZToMuMuRECOEventContent.outputCommands.extend(zToMuMuRECOEventContent.outputCommands)

