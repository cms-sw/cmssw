import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
zToTauTau_DoubleTauOutputModule = cms.OutputModule("PoolOutputModule",
    AODSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToTauTau_DoubleTau'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToTauTau_DoubleTauHLTPath')
    ),
    fileName = cms.untracked.string('zToTauTau_DoubleTau.root')
)


