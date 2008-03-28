import FWCore.ParameterSet.Config as cms

from QCDAnalysis.Skimming.diMuonEventContent_cfi import *
diMuonOutputModule = cms.OutputModule("PoolOutputModule",
    diMuonEventSelection,
    diMuonEventContent,
    dataset = cms.untracked.PSet(
        filterName = cms.untracked.string('diMuons'),
        dataTier = cms.untracked.string('USER')
    ),
    fileName = cms.untracked.string('diMuons.root')
)


