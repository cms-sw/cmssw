import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
wToMuNuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('wToMuNu_Filter'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('wToMuNuHLTPath')
    ),
    fileName = cms.untracked.string('wToMuNu.root')
)


