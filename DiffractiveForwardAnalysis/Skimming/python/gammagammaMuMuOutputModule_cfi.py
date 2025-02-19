import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
gammagammaMuMuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('gammagammaMuMu'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('gammagammaMuMuHLTPath')
    ),
    fileName = cms.untracked.string('gammagammaMuMuFiltered.root')
)


