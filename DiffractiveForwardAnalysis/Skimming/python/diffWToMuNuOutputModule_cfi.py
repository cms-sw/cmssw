import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
diffWToMuNuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('diffWToMuNu'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('diffWToMuNuHLTPath')
    ),
    fileName = cms.untracked.string('diffWToMuNuFiltered.root')
)


