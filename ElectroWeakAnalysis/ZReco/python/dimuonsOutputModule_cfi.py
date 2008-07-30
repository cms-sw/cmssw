import FWCore.ParameterSet.Config as cms

from ElectroWeakAnalysis.ZReco.dimuons_EventContent_cff import *
from Configuration.EventContent.EventContent_cff import *
AODSIMDimuonEventContent = cms.PSet(
    outputCommands = cms.untracked.vstring()
)
dimuonsOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMDimuonEventContent,
    dimuonsEventSelection,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('dimuon'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('dimuons.root')
)

AODSIMDimuonEventContent.outputCommands.extend(AODSIMEventContent.outputCommands)
AODSIMDimuonEventContent.outputCommands.extend(dimuonsEventContent.outputCommands)

