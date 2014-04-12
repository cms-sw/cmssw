import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
gammagammaEEOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('gammagammaEE'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('gammagammaEEHLTPath')
    ),
    fileName = cms.untracked.string('gammagammaEEFiltered.root')
)


