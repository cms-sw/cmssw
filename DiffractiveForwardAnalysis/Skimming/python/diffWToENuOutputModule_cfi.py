import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
diffWToENuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('diffWToENu'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('diffWToENuHLTPath')
    ),
    fileName = cms.untracked.string('diffWToENuFiltered.root')
)


