import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
zToTauTau_EMuOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToTauTau_EMu'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToTauTau_EMuHLTPath')
    ),
    fileName = cms.untracked.string('zToTauTau_EMu.root')
)


