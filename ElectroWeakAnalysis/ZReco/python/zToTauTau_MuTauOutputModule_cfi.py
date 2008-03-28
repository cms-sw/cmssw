import FWCore.ParameterSet.Config as cms

from Configuration.EventContent.EventContent_cff import *
zToTauTau_MuTauOutputModule = cms.OutputModule("PoolOutputModule",
    RECOSIMEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('zToTauTau_MuTau'),
        dataTier = cms.untracked.string('USER')
    ),
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('zToTauTau_MuTauHLTPath')
    ),
    fileName = cms.untracked.string('zToTauTau_MuTau.root')
)


