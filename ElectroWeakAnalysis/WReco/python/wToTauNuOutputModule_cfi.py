import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
wToTauNuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('wToTauNu_Filter'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('wToTauNuHLTPath')
    ),
    fileName = cms.untracked.string('wToTauNu.root')
)


